#include "tts_backend.h"

#include "omni.h"

namespace {

class LegacyTtsBackend : public OmniTtsBackend {
public:
    const char * name() const override {
        return "legacy";
    }

    bool init(omni_context * ctx, const OmniTtsBackendOptions & options, std::string & error) override {
        if (ctx == nullptr) {
            error = "omni context is null";
            return false;
        }
        ctx->tts_bin_dir = options.legacy_tts_bin_dir;
        return true;
    }

    void reset(omni_context * ctx) override {
        if (ctx == nullptr) {
            return;
        }
        ctx->tts_token_buffer.clear();
        ctx->tts_all_generated_tokens.clear();
        ctx->tts_n_past_accumulated = 0;
        ctx->tts_condition_saved = false;
    }

    bool synthesize_text(omni_context *, const std::string &, const std::string &, bool, std::string &) override {
        return false;
    }
};

} // namespace

std::unique_ptr<OmniTtsBackend> create_omni_tts_backend() {
    return std::make_unique<LegacyTtsBackend>();
}

const char * omni_tts_backend_name() {
    return "legacy";
}

