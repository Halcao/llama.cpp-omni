#pragma once

#include <memory>
#include <string>

struct common_params;
struct omni_context;

struct OmniTtsBackendOptions {
    std::string model_path;
    std::string legacy_tts_bin_dir;
    std::string device = "cpu";
    int n_threads = 4;
    int gpu_layers = -1;
};

class OmniTtsBackend {
public:
    virtual ~OmniTtsBackend() = default;

    virtual const char * name() const = 0;
    virtual bool init(omni_context * ctx, const OmniTtsBackendOptions & options, std::string & error) = 0;
    virtual void reset(omni_context * ctx) = 0;

    // Returns true only when the backend produced a wav file directly.
    virtual bool synthesize_text(omni_context * ctx,
                                 const std::string & text,
                                 const std::string & output_wav_path,
                                 bool is_final,
                                 std::string & error) = 0;
};

std::unique_ptr<OmniTtsBackend> create_omni_tts_backend();
const char * omni_tts_backend_name();

