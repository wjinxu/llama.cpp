#pragma once

#include "models.h"

// shared scaffolding for DFlash-style block-diffusion drafts (see dflash-base.cpp):
// encoder fusion, draft-token/lm_head plumbing via ctx_other and the DSpark markov head
struct llm_build_dflash_base : public llm_graph_context {
    llm_build_dflash_base(const llm_graph_params & params);

    ggml_tensor * build_dflash_inp_embd_enc();
    void          build_dflash_encoder(const llama_model_dflash & model);

    ggml_tensor * build_dflash_inp_embd();
    ggml_tensor * build_dflash_inp_tokens(const llama_model_dflash & model);

    void build_dflash_head(const llama_model_dflash & model, ggml_tensor * cur);

    void build_dspark_markov_head(const llama_model_dflash & model);

    ggml_tensor * inp_tokens = nullptr;
};

// the hooks of one DFlash type (backbone family), implemented in the backbone's own
// model file. name matches the GGUF `dflash.dflash_type` string written by the converter; the
// loader defaults to "qwen3" when the key is absent, so checkpoints predating it still load
struct llm_dflash_type {
    const char * name;

    void (*load_hparams)(llama_model_dflash & model, llama_model_loader & ml);
    void (*load_tensors)(llama_model_dflash & model);

    std::unique_ptr<llm_graph_context> (*build_graph)(const llama_model_dflash & model, const llm_graph_params & params);
};

// one entry per DFlash type, listed in dflash-base.cpp
extern const llm_dflash_type llm_dflash_type_qwen3;
