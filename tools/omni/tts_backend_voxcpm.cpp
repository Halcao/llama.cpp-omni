#include "tts_backend.h"

#include "omni.h"

#include "voxcpm/audio_io.h"
#include "voxcpm/backend.h"
#include "voxcpm/server_common.h"

#include <exception>
#include <fstream>
#include <memory>

namespace {

voxcpm::BackendType parse_backend_type(const std::string & device) {
    if (device.rfind("gpu", 0) == 0 || device.rfind("cuda", 0) == 0) {
        return voxcpm::BackendType::CUDA;
    }
    if (device.rfind("metal", 0) == 0) {
        return voxcpm::BackendType::Metal;
    }
    if (device.rfind("vulkan", 0) == 0) {
        return voxcpm::BackendType::Vulkan;
    }
    if (device.rfind("auto", 0) == 0) {
        return voxcpm::BackendType::Auto;
    }
    return voxcpm::BackendType::CPU;
}

class VoxcpmTtsBackend : public OmniTtsBackend {
public:
    const char * name() const override {
        return "voxcpm";
    }

    bool init(omni_context *, const OmniTtsBackendOptions & options, std::string & error) override {
        if (options.model_path.empty()) {
            error = "VoxCPM model path is empty";
            return false;
        }

        core_ = std::make_unique<voxcpm::VoxCPMServiceCore>(
                options.model_path, parse_backend_type(options.device), options.n_threads);
        try {
            core_->load();
        } catch (const std::exception & ex) {
            error = std::string("failed to initialize VoxCPM service core: ") + ex.what();
            return false;
        }

        return true;
    }

    void reset(omni_context *) override {
    }

    bool synthesize_text(omni_context *,
                         const std::string & text,
                         const std::string & output_wav_path,
                         bool,
                         std::string & error) override {
        if (!core_) {
            error = "VoxCPM backend is not initialized";
            return false;
        }
        if (text.empty()) {
            return true;
        }

        try {
            voxcpm::SynthesisRequest request;
            request.text = text;
            request.prompt.patch_size = core_->patch_size();
            request.prompt.feat_dim = core_->feat_dim();
            request.prompt.sample_rate = core_->sample_rate();
            voxcpm::SynthesisResult result = core_->synthesize(request);
            std::vector<uint8_t> wav = voxcpm::encode_audio(voxcpm::AudioResponseFormat::Wav,
                                                            result.waveform,
                                                            result.sample_rate);

            std::ofstream out(output_wav_path, std::ios::binary);
            if (!out.is_open()) {
                error = "failed to open VoxCPM wav output: " + output_wav_path;
                return false;
            }
            out.write(reinterpret_cast<const char *>(wav.data()), static_cast<std::streamsize>(wav.size()));
            return out.good();
        } catch (const std::exception & ex) {
            error = std::string("VoxCPM synthesis failed: ") + ex.what();
            return false;
        }
    }

private:
    std::unique_ptr<voxcpm::VoxCPMServiceCore> core_;
};

} // namespace

std::unique_ptr<OmniTtsBackend> create_omni_tts_backend() {
    return std::make_unique<VoxcpmTtsBackend>();
}

const char * omni_tts_backend_name() {
    return "voxcpm";
}

