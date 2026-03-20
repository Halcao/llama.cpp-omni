#include <ggml-cuda.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kNumBits = 4;
constexpr int kTile = 16;
constexpr int kPackFactor = 32 / kNumBits;

void check_cuda(cudaError_t err, const char * expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(expr) + ": " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

std::vector<int> get_weight_perm() {
    std::vector<int> perm_list;
    perm_list.reserve(1024);

    for (int i = 0; i < 32; ++i) {
        std::vector<int> perm1;
        perm1.reserve(8);
        const int col = i / 4;
        for (int block : {0, 1}) {
            for (int row : {2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1}) {
                perm1.push_back(16 * row + col + 8 * block);
            }
        }
        for (int j = 0; j < 4; ++j) {
            for (int p : perm1) {
                perm_list.push_back(p + 256 * j);
            }
        }
    }

    const int interleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    std::vector<int> perm;
    perm.reserve(perm_list.size());
    for (size_t i = 0; i < perm_list.size(); i += 8) {
        for (int idx : interleave) {
            perm.push_back(perm_list[i + idx]);
        }
    }

    return perm;
}

std::vector<int> get_scale_perm_single() {
    std::vector<int> perm;
    perm.reserve(32);
    for (int i = 0; i < 4; ++i) {
        for (int j : {0, 1, 8, 9, 16, 17, 24, 25}) {
            perm.push_back(2 * i + j);
        }
    }
    return perm;
}

std::vector<uint32_t> marlin_pack_u4_weights(const std::vector<uint32_t> & q_w, int size_k, int size_n) {
    const auto perm = get_weight_perm();
    const int tile_k_blocks = size_k / kTile;
    const int tile_n_blocks = size_n / kTile;

    std::vector<uint32_t> tiled(tile_k_blocks * size_n * kTile);
    for (int kb = 0; kb < tile_k_blocks; ++kb) {
        for (int nb = 0; nb < tile_n_blocks; ++nb) {
            for (int ki = 0; ki < kTile; ++ki) {
                for (int ni = 0; ni < kTile; ++ni) {
                    const int src_row = kb * kTile + ki;
                    const int src_col = nb * kTile + ni;
                    const int dst_col = nb * kTile * kTile + ki * kTile + ni;
                    tiled[kb * (size_n * kTile) + dst_col] = q_w[src_row * size_n + src_col];
                }
            }
        }
    }

    std::vector<uint32_t> permuted(tiled.size());
    const int perm_width = static_cast<int>(perm.size());
    for (size_t base = 0; base < tiled.size(); base += perm_width) {
        for (int i = 0; i < perm_width; ++i) {
            permuted[base + i] = tiled[base + perm[i]];
        }
    }

    std::vector<uint32_t> packed(tile_k_blocks * (size_n * kTile / kPackFactor), 0);
    const int packed_cols = size_n * kTile / kPackFactor;
    for (int row = 0; row < tile_k_blocks; ++row) {
        for (int col = 0; col < size_n * kTile; ++col) {
            const int packed_col = col / kPackFactor;
            const int shift = (col % kPackFactor) * kNumBits;
            packed[row * packed_cols + packed_col] |= (permuted[row * (size_n * kTile) + col] & 0xFu) << shift;
        }
    }

    return packed;
}

std::vector<__half> marlin_permute_scales_single(const std::vector<__half> & scales, int size_n) {
    const auto perm = get_scale_perm_single();
    std::vector<__half> out(scales.size());
    const int block = static_cast<int>(perm.size());
    for (int row = 0; row < static_cast<int>(scales.size()) / size_n; ++row) {
        const __half * src = scales.data() + row * size_n;
        __half * dst = out.data() + row * size_n;
        for (int base = 0; base < size_n; base += block) {
            for (int i = 0; i < block; ++i) {
                dst[base + i] = src[base + perm[i]];
            }
        }
    }
    return out;
}

float ref_weight(uint32_t q) {
    return static_cast<float>(static_cast<int>(q) - 8);
}

} // namespace

int main() {
    int device_count = 0;
    cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: CUDA device not available\n";
        return 0;
    }

    constexpr int m = 16;
    constexpr int n = 64;
    constexpr int k = 128;
    constexpr int group_size = -1;
    constexpr int num_groups = 1;

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major * 10 + minor < 75) {
        std::cout << "SKIP: Marlin requires compute capability >= 7.5\n";
        return 0;
    }

    std::vector<__half> a_host(m * k);
    std::vector<uint32_t> q_w_host(k * n);
    std::vector<__half> scales_host(num_groups * n, __float2half(1.0f));
    std::vector<float> ref_host(m * n, 0.0f);

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            const float value = float(((row * 5 + col * 3) % 11) - 5) * 0.5f;
            a_host[row * k + col] = __float2half(value);
        }
    }

    for (int row = 0; row < k; ++row) {
        for (int col = 0; col < n; ++col) {
            q_w_host[row * n + col] = static_cast<uint32_t>((row * 7 + col * 5 + 3) & 0xF);
        }
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += __half2float(a_host[row * k + kk]) * ref_weight(q_w_host[kk * n + col]);
            }
            ref_host[row * n + col] = acc;
        }
    }

    const std::vector<uint32_t> packed_b_host = marlin_pack_u4_weights(q_w_host, k, n);
    const std::vector<__half> packed_scales_host = marlin_permute_scales_single(scales_host, n);

    __half * d_a = nullptr;
    uint32_t * d_b = nullptr;
    __half * d_scales = nullptr;
    __half * d_c = nullptr;
    int * d_workspace = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), a_host.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), packed_b_host.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_scales), packed_scales_host.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), m * n * sizeof(__half)));

    const int workspace_elems = ggml_cuda_marlin_min_workspace_elements(device);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_workspace), workspace_elems * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, a_host.data(), a_host.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, packed_b_host.data(), packed_b_host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, packed_scales_host.data(), packed_scales_host.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, m * n * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_workspace, 0, workspace_elems * sizeof(int)));

    ggml_init_params ggml_params = {
        /*.mem_size   =*/ 1 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ggml_params);
    if (ctx == nullptr) {
        std::cerr << "ggml_init failed\n";
        return 1;
    }

    ggml_tensor * a_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    ggml_tensor * b_qweight_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, n / kPackFactor);
    ggml_tensor * b_scales_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, num_groups, n);
    ggml_tensor * c_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, m);
    ggml_tensor * workspace_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, workspace_elems);

    a_tensor->data = d_a;
    b_qweight_tensor->data = d_b;
    b_scales_tensor->data = d_scales;
    c_tensor->data = d_c;
    workspace_tensor->data = d_workspace;

    if (!ggml_cuda_marlin_w4a16_gemm(
                a_tensor,
                b_qweight_tensor,
                b_scales_tensor,
                nullptr,
                c_tensor,
                workspace_tensor,
                device,
                nullptr)) {
        std::cerr << "ggml_cuda_marlin_w4a16_gemm returned false\n";
        ggml_free(ctx);
        return 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> c_host(m * n);
    CUDA_CHECK(cudaMemcpy(c_host.data(), d_c, c_host.size() * sizeof(__half), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        const float got = __half2float(c_host[i]);
        const float want = ref_host[i];
        max_abs_err = std::max(max_abs_err, std::fabs(got - want));
        if (std::fabs(got - want) > 1e-2f) {
            std::cerr << "Mismatch at " << i << ": got=" << got << " want=" << want << '\n';
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
    ggml_free(ctx);

    std::cout << "Marlin W4A16 GEMM test passed, max_abs_err=" << max_abs_err << '\n';
    return 0;
}
