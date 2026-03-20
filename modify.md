# AWQ Qwen3 接入 Marlin 修改说明

这份文档总结当前代码状态下，要让 `Qwen3 + AWQ` 在推理图中把所有 linear 都切到 Marlin 实现，需要修改哪些代码、为什么要改、以及建议怎么改。

当前已经完成的部分：

- GGUF 元信息读取：
  - `quant_method` 已经改成 enum，定义在 `src/llama-hparams.h`
  - `quant_bits` / `quant_group_size` / `quant_zero_point` 已经读入 `hparams`
- Qwen3 AWQ tensor 注册：
  - `qweight` / `qzeros` / `scales` 的名字映射已经加到 `src/llama-arch.h` / `src/llama-arch.cpp`
  - `llama_layer` 中已经有 `wq_awq_qweight` 等字段，定义在 `src/llama-model.h`
  - `QWEN3` 分支中已经会创建这些 tensor，代码在 `src/llama-model.cpp`
- CUDA 侧基础接口：
  - 已经有 `ggml_cuda_marlin_awq_repack(...)`
  - 已经有 `ggml_cuda_marlin_w4a16_gemm(...)`
  - `test-marlin-w4a16` 已通过，说明底层单次 GEMM 接口可用

当前还没完成的核心部分：

- 推理图里还没有“AWQ linear 节点”
- `llm_build_qwen3` 里所有 linear 仍然通过 `ggml_mul_mat`
- CUDA backend 还不知道如何执行“AWQ Marlin 节点”
- AWQ 原始 GGUF 权重还没有转换成 Marlin 需要的布局

所以要完成的不是一个点改动，而是一条完整链路：

1. 模型加载阶段把 AWQ tensor 读入
2. 必要时把 AWQ 原始布局 repack/permutate 成 Marlin 布局
3. 图构建阶段为 AWQ linear 生成专门节点
4. CUDA backend 执行该节点时调用 `ggml_cuda_marlin_w4a16_gemm`

---

## 一、Qwen3 中哪些 linear 要替换成 Marlin

文件：

- `src/llama-model.cpp`
- 结构体：`llm_build_qwen3`

函数位置：

- `llm_build_qwen3`

当前代码中需要替换的线性层：

1. Attention Q projection
   - 当前：`build_lora_mm(model.layers[il].wq, cur)`
2. Attention K projection
   - 当前：`build_lora_mm(model.layers[il].wk, cur)`
3. Attention V projection
   - 当前：`build_lora_mm(model.layers[il].wv, cur)`
4. Attention output projection
   - 当前通过 `build_attn(..., model.layers[il].wo, ...)` 内部的 `build_lora_mm(wo, cur)` 完成
5. FFN up projection
   - 当前通过 `build_ffn(...)` 内部 `build_lora_mm(up, cur)` 完成
6. FFN gate projection
   - 当前通过 `build_ffn(...)` 内部 `build_lora_mm(gate, cur)` 完成
7. FFN down projection
   - 当前通过 `build_ffn(...)` 内部 `build_lora_mm(down, cur)` 完成

说明：

- 这 7 个才是 Qwen3 主干里最关键的 linear
- 最后的 lm_head 目前还是 `model.output`
- 你现在没有为 `output` 增加 AWQ tensor，所以第一阶段可以先不处理 lm_head

为什么这里必须改：

- 现在 `llm_build_qwen3` 图构建时，linear 全都被抽象成 `build_lora_mm(...)`
- `build_lora_mm(...)` 内部直接调用 `ggml_mul_mat(...)`
- 所以即使 AWQ tensor 已经被加载，推理图也完全不会使用它们

建议修改方式：

- 在 `llm_build_qwen3` 中对 Q、K、V、O、FFN 三个线性显式分流
- 如果当前层满足：
  - `hparams.quant_method == LLAMA_QUANTIZATION_METHOD_AWQ`
  - 且对应 `*_awq_qweight` / `*_awq_scale` 非空
- 则走新的 AWQ Marlin helper
- 否则继续走现有 `build_lora_mm(...)`

注意：

- 不建议直接在一开始就全局改 `build_lora_mm(...)` 为“自动识别 AWQ”
- 因为 `build_lora_mm(...)` 是所有架构共用入口，容易把别的模型路径带坏
- 第一阶段建议只在 `llm_build_qwen3` 做显式分流

---

## 二、需要新增一个 AWQ 专用的 graph helper

文件：

- `src/llama-graph.h`
- `src/llama-graph.cpp`

当前相关函数：

- `llm_graph_context::build_lora_mm(...)`
- `llm_graph_context::build_ffn(...)`
- `llm_graph_context::build_attn(...)`

为什么这里必须改：

- `llm_build_qwen3` 只负责拼模型结构
- 真正的“linear 节点怎么建”是在 graph helper 中封装的
- 当前 helper 只能创建普通 `ggml_mul_mat`
- 没有任何一个 helper 能表达：
  - 输入激活 `a`
  - AWQ `qweight`
  - AWQ `scales`
  - AWQ `qzeros`
  - workspace
  - 然后把这几个拼成一个“Marlin linear 节点”

建议新增的 helper：

- `build_awq_marlin_mm(...)`

建议签名示例：

```cpp
ggml_tensor * llm_graph_context::build_awq_marlin_mm(
    ggml_tensor * cur,
    ggml_tensor * qweight,
    ggml_tensor * qzeros,
    ggml_tensor * scales,
    int64_t out_features,
    int il) const;
```

这个 helper 需要负责：

1. 检查 `qweight/scales` 是否存在
2. 为输出创建目标 tensor
3. 创建或获取 workspace tensor
4. 构造新的 GGML op 节点，例如 `ggml_marlin_w4a16(...)`
5. 返回结果 tensor，供后续 `reshape/norm/add` 使用

为什么不建议继续复用 `build_lora_mm(...)`：

- `build_lora_mm(...)` 目前只接收一个权重 tensor `w`
- AWQ/Marlin 实际需要 3 到 5 个输入
- 如果硬往 `build_lora_mm(...)` 里塞，会让通用接口语义变乱

建议：

- 保留 `build_lora_mm(...)` 作为普通 dense 路径
- 新增 `build_awq_marlin_mm(...)` 作为 AWQ 路径

---

## 三、Qwen3 attention output 和 FFN 不能完全依赖现有公共 helper

文件：

- `src/llama-model.cpp`
- `src/llama-graph.cpp`

为什么这点要单独说：

- `Q/K/V` 比较容易，直接在 `llm_build_qwen3` 里把
  - `build_lora_mm(wq, cur)`
  - 替换成
  - `build_awq_marlin_mm(cur, wq_awq_qweight, ..., out_dim, il)`
- 但是 `wo/up/gate/down` 现在藏在公共 helper 里

具体情况：

1. `wo`
   - 在 `build_attn(...)` 内部统一执行
   - 当前接口只接受 `wo` 和 `wo_b`
   - 没法把 `wo_awq_qweight/wo_awq_qzeros/wo_awq_scale` 传进去

2. `ffn_up/gate/down`
   - 在 `build_ffn(...)` 内部统一执行
   - 当前也只接受普通 dense tensor
   - 没法传 AWQ 三元组

建议修改方式：

### 方案 A：Qwen3 局部改开，最快最稳

在 `llm_build_qwen3` 中：

- attention 先只构建 `Q/K/V`
- 调用一个不带 `wo` 的 attention helper，只返回 attention 输出
- 然后自己在外面做 `wo`
  - AWQ 时走 `build_awq_marlin_mm`
  - 非 AWQ 时走 `build_lora_mm`

FFN 同理：

- 不再直接调用通用 `build_ffn(...)`
- 在 `llm_build_qwen3` 里把 up/gate/down 三个线性手工展开
- 分别按 AWQ / 非 AWQ 选择不同 helper

优点：

- 改动局部
- 不会影响其它架构
- 最适合第一版

缺点：

- `llm_build_qwen3` 会比现在更长

### 方案 B：扩展通用 helper

扩展：

- `build_attn(...)`
- `build_ffn(...)`

让它们能接收 AWQ 三元组或一个新的 linear descriptor

优点：

- 结构更优雅

缺点：

- 改动面明显更大
- 会影响很多架构共用逻辑

结论：

- 第一版建议选方案 A

---

## 四、必须在 GGML 里增加一个新的 Marlin op

文件：

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`

为什么这里必须改：

- 现在 graph 里只有 `GGML_OP_MUL_MAT`
- `ggml_cuda_marlin_w4a16_gemm(...)` 只是 CUDA 侧一个裸函数
- 如果不新增 op，图执行器根本不知道什么时候调用它

建议新增：

1. 新 op 类型

例如：

- `GGML_OP_MARLIN_W4A16`

2. 新建图接口

例如：

```cpp
struct ggml_tensor * ggml_marlin_w4a16(
    struct ggml_context * ctx,
    struct ggml_tensor * a,
    struct ggml_tensor * qweight,
    struct ggml_tensor * scales,
    struct ggml_tensor * qzeros,
    struct ggml_tensor * workspace);
```

这个函数需要做的事：

- 设置 `dst->op`
- 填充 `dst->src[0..n]`
- 设置输出 shape
- 必要时设置 op params

还需要同步补：

- op name / debug string
- shape 推导
- 如果 GGML 有相关 op 校验逻辑，也要补

为什么不建议直接用 `ggml_map_custom3(...)`：

- 可以作为临时方案，但长期维护性差
- CUDA backend 里对 custom op 的调度支持并不天然等同于一个正式 op
- 你现在是要把 Marlin 接成正式推理路径，建议直接加正式 op

---

## 五、CUDA backend 要识别并执行新的 AWQ Marlin op

文件：

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- 可能新增：
  - `ggml/src/ggml-cuda/marlin-op.cu`
  - `ggml/src/ggml-cuda/marlin-op.cuh`

为什么这里必须改：

- GGML graph 节点最终是由 backend 执行
- CUDA backend 现在只认识已有 op
- 即便你在 graph 里建出 `GGML_OP_MARLIN_W4A16`，如果 backend 不处理，它也不会运行

建议修改点：

1. 在 `switch (node->op)` 里加入新分支
2. 提取：
   - `a`
   - `qweight`
   - `scales`
   - `qzeros`
   - `workspace`
   - `dst`
3. 调用：

```cpp
ggml_cuda_marlin_w4a16_gemm(...)
```

建议另外封装一个 backend 层 helper：

```cpp
void ggml_cuda_op_marlin_w4a16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

放在单独文件里更清楚，方便以后再接 GPTQ Marlin。

---

## 六、AWQ 权重不能直接拿原始 GGUF 布局喂 Marlin

文件：

- `ggml/include/ggml-cuda.h`
- `ggml/src/ggml-cuda/awq_marlin_repack.cu`
- `src/llama-model.cpp`
- `src/llama-model-loader.cpp`

为什么这里必须改：

- 现在从 GGUF 读进来的 AWQ tensor 是 HuggingFace/AWQ 原始格式
- Marlin kernel 需要的是它自己的 pack/reorder 后格式
- 你的测试 `test-marlin-w4a16.cpp` 里已经手工做过：
  - weight pack
  - scale permutation

当前已有基础：

- `ggml_cuda_marlin_awq_repack(...)` 已存在
- 但它还没有接入模型加载主流程

这里至少要解决两件事：

1. `qweight` repack
2. `scales` permutation

注意：

- 你现在的 `ggml_cuda_marlin_w4a16_gemm(...)` 直接拿的是“已经处理好布局”的 `b_q_weight` / `b_scales`
- 也就是说，repack/permutation 必须在它之前完成

建议做法：

### 方案 A：加载后立即转成 Marlin 布局

优点：

- 推理图最简单
- 每次执行不用再判断

缺点：

- 加载逻辑更复杂

建议步骤：

1. `load_all_data(...)` 完成后
2. 检测 `Qwen3 + AWQ`
3. 对每层的 `wq/wk/wv/wo/ffn_up/ffn_gate/ffn_down`
   - 读取原始 AWQ tensor
   - 调用 CUDA repack
   - 生成 Marlin layout tensor

### 方案 B：第一次使用时懒惰转换

优点：

- 加载阶段简单

缺点：

- 推理首 token 逻辑更复杂
- 需要缓存 repack 后结果

第一版建议：

- 优先方案 A

此外你还要决定 Marlin 格式存哪里：

1. 直接覆写原 `qweight/scales`
2. 新增字段存“repacked”版本

建议新增字段，最清楚，例如：

- `wq_awq_marlin_qweight`
- `wq_awq_marlin_scale`

否则很难分辨当前 tensor 是“原始 GGUF 格式”还是“Marlin 格式”。

---

## 七、需要给 Marlin op 提供 workspace

文件：

- `ggml/include/ggml-cuda.h`
- 新的 GGML Marlin op 实现
- `src/llama-graph.cpp`

为什么这里必须改：

- `ggml_cuda_marlin_w4a16_gemm(...)` 明确要求 workspace
- 测试里 workspace 是显式分配的
- 推理图里如果没有 workspace，Marlin op 无法调用

建议方式：

### 方案 A：graph 中显式建 workspace tensor

比如在 `build_awq_marlin_mm(...)` 里：

- 根据当前 device 对应最小 workspace size 建一个 tensor
- 把它作为 Marlin op 的一个输入

优点：

- 图语义明确

缺点：

- 需要 device 信息或保守大小

### 方案 B：CUDA backend 内部缓存 workspace

优点：

- graph 更简洁

缺点：

- backend 状态更复杂
- 多张图/多 stream 时更容易踩坑

第一版建议：

- 先走方案 B 或简化版缓存
- 后续再优化成 graph 显式 workspace

---

## 八、建议的最小改动顺序

为了尽快做出第一版“Qwen3 AWQ 能跑通”的实现，建议按下面顺序改：

### 第 1 步：先补 GGML 正式 op

文件：

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`

目标：

- 新增 `GGML_OP_MARLIN_W4A16`
- 新增 `ggml_marlin_w4a16(...)`

### 第 2 步：接 CUDA backend

文件：

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- 新增 Marlin op 执行文件

目标：

- 新 op 被 CUDA backend 识别
- 能调用 `ggml_cuda_marlin_w4a16_gemm(...)`

### 第 3 步：新增 graph helper

文件：

- `src/llama-graph.h`
- `src/llama-graph.cpp`

目标：

- 新增 `build_awq_marlin_mm(...)`

### 第 4 步：改 Qwen3 builder

文件：

- `src/llama-model.cpp`

目标：

- 在 `llm_build_qwen3` 中对以下线性显式分流：
  - `wq`
  - `wk`
  - `wv`
  - `wo`
  - `ffn_up`
  - `ffn_gate`
  - `ffn_down`

### 第 5 步：加入 AWQ -> Marlin repack 流程

文件：

- `src/llama-model.cpp`
- `src/llama-model-loader.cpp`
- 或加载后初始化阶段的合适位置

目标：

- 模型加载结束后，生成 Marlin 可直接消费的权重布局

### 第 6 步：补验证

建议至少加：

1. 单层 linear 测试
   - 用真实 Qwen3 AWQ tensor 走一次 Marlin
2. 图构建测试
   - 确认 `Qwen3 + AWQ` 命中了新 op，而不是 `ggml_mul_mat`
3. 小模型推理 smoke test

---

## 九、每个文件建议修改摘要

### 1. `src/llama-model.cpp`

要改的地方：

- `llm_build_qwen3`

为什么：

- 这是 Qwen3 主图入口
- 这里决定每个 linear 是走 dense 还是走 AWQ/Marlin

怎么改：

- 对 Q/K/V 直接改成条件分流
- 对 `wo` 和 `ffn_*` 不要再完全依赖现有通用 helper
- 在这里显式调用新的 `build_awq_marlin_mm(...)`

### 2. `src/llama-graph.h`

要改的地方：

- `llm_graph_context` 声明

为什么：

- 要新增 AWQ/Marlin 专用 helper

怎么改：

- 增加 `build_awq_marlin_mm(...)` 声明

### 3. `src/llama-graph.cpp`

要改的地方：

- `build_lora_mm(...)` 附近

为什么：

- 当前这里只有普通 `ggml_mul_mat`

怎么改：

- 新增 `build_awq_marlin_mm(...)`
- 必要时增加 workspace 获取逻辑

### 4. `ggml/include/ggml.h`

要改的地方：

- op enum
- 新 op API 声明

为什么：

- graph 层必须能表达 Marlin linear

怎么改：

- 增加 `GGML_OP_MARLIN_W4A16`
- 增加 `ggml_marlin_w4a16(...)`

### 5. `ggml/src/ggml.c`

要改的地方：

- 新 op 创建逻辑
- 新 op 名字/参数/shape

为什么：

- graph 节点要能真正建出来

怎么改：

- 实现 `ggml_marlin_w4a16(...)`

### 6. `ggml/src/ggml-cuda/ggml-cuda.cu`

要改的地方：

- backend op dispatch

为什么：

- CUDA backend 要认识新 op

怎么改：

- 新增 `case GGML_OP_MARLIN_W4A16:`
- 在里面调用新的 `ggml_cuda_op_marlin_w4a16(...)`

### 7. 新增 CUDA 执行文件

建议文件：

- `ggml/src/ggml-cuda/marlin-op.cu`
- `ggml/src/ggml-cuda/marlin-op.cuh`

为什么：

- 不建议把 Marlin op 全塞进 `ggml-cuda.cu`

怎么改：

- 封装 `ggml_cuda_op_marlin_w4a16(...)`
- 从 `dst->src[]` 组装调用参数
- 调 `ggml_cuda_marlin_w4a16_gemm(...)`

### 8. `src/llama-model-loader.cpp` 或加载后阶段

为什么：

- AWQ 原始权重还不能直接给 Marlin 用

怎么改：

- 在合适阶段完成 AWQ 到 Marlin 布局转换

---

## 十、实现时的几个关键注意点

### 1. 不能只改 `llm_build_qwen3`

原因：

- 只改 builder 但没有新 GGML op，最后还是无法执行 Marlin

### 2. 不能只加新 op 而不做 repack

原因：

- AWQ 原始 `qweight/scales` 布局不等于 Marlin 输入布局

### 3. `wo` 和 `ffn_*` 是最容易漏的

原因：

- 它们不是在 `llm_build_qwen3` 里直接写 `ggml_mul_mat`
- 而是藏在公共 helper 里

### 4. 第一版不要试图全架构通用化

原因：

- 现在目标很明确，就是 `Qwen3 + AWQ + Marlin`
- 先把一条路径打通，比一开始做成全模型通用更稳

---

## 十一、推荐的第一版目标

建议把第一版范围控制为：

- 仅支持 `LLM_ARCH_QWEN3`
- 仅支持 `LLAMA_QUANTIZATION_METHOD_AWQ`
- 仅支持 CUDA backend
- 仅支持 W4A16
- 仅替换 Qwen3 主干 7 个 linear
- 不处理 lm_head

这样可以最快得到一个“真正能跑”的版本。

等这版跑通后，再考虑：

- GPTQ Marlin
- lm_head AWQ
- 通用化到其它架构
- 更优雅的 helper 抽象

---

## 十二、最后的实现判断标准

你做完后，至少应满足下面几个判断标准：

1. `Qwen3 + 非 AWQ`
   - 仍走原来的 `ggml_mul_mat`
2. `Qwen3 + AWQ`
   - Q/K/V/O/FFN 都命中新 AWQ Marlin 路径
3. CUDA graph 执行时
   - 真正调用到 `ggml_cuda_marlin_w4a16_gemm(...)`
4. 权重布局
   - 不是直接把原始 GGUF AWQ tensor 生喂给 Marlin
5. 数值正确性
   - 至少有一组对照测试

如果要继续推进实现，建议下一步直接按这份文档的“最小改动顺序”动手。
