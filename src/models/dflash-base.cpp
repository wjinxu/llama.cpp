#include "dflash-base.h"

#include "llama-impl.h"

static const llm_dflash_type * dflash_types[] = {
    &llm_dflash_type_qwen3,
};

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    std::string type_name = "qwen3"; // default for checkpoints predating the metadata key
    ml.get_key(LLM_KV_DFLASH_TYPE, type_name, false);

    for (const auto * t : dflash_types) {
        if (type_name == t->name) {
            dflash_type = t;
            break;
        }
    }
    if (dflash_type == nullptr) {
        throw std::runtime_error("unknown DFlash type '" + type_name + "'");
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_BLOCK_SIZE, block_size, false);

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    LLAMA_LOG_INFO("%s: DFlash type = %s\n", __func__, dflash_type->name);

    LLAMA_LOG_INFO("%s: DFlash extract_layers = [", __func__);
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        LLAMA_LOG_INFO("%d%s", target_layer_ids[i], i + 1 < target_layer_ids.size() ? ", " : "");
    }
    LLAMA_LOG_INFO("]\n");

    dflash_type->load_hparams(*this, ml);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        if (block_size == 0) {
            throw std::runtime_error("DSpark draft requires 'dflash.block_size' in GGUF metadata");
        }

        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab }, 0);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, 0);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    dflash_type->load_tensors(*this);
}

// DFlash Encoder: processes target model features through feature fusion layer
struct llm_build_dflash_enc : public llm_build_dflash_base {
    llm_build_dflash_enc(const llama_model_dflash & model, const llm_graph_params & params) : llm_build_dflash_base(params) {
        build_dflash_encoder(model);
    }
};

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<llm_build_dflash_enc>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return dflash_type->build_graph(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    };
}

llm_build_dflash_base::llm_build_dflash_base(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_build_dflash_base::build_dflash_inp_embd_enc() {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

void llm_build_dflash_base::build_dflash_encoder(const llama_model_dflash & model) {
    ggml_tensor * cur = build_dflash_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llm_build_dflash_base::build_dflash_inp_embd() {
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * inp_g = inp->embd;
    cb(inp_g, "inp_g_embeddings", -1);

    res->add_input(std::move(inp));

    return inp_g;
}

ggml_tensor * llm_build_dflash_base::build_dflash_inp_tokens(const llama_model_dflash & model) {
    // tok_embd from the target model (shared via ctx_other)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    return inpL;
}

void llm_build_dflash_base::build_dflash_head(const llama_model_dflash & model, ggml_tensor * cur) {
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model (shared via ctx_other)
    auto * output = model.output;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output = model_other->output;
    }

    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        build_dspark_markov_head(model);
    }
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
void llm_build_dflash_base::build_dspark_markov_head(const llama_model_dflash & model) {
    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && model.dspark_conf_proj && "DSpark markov/confidence weights not loaded");
    GGML_ASSERT(inp_tokens && "DSpark markov head requires the token inputs");

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const int64_t block_size = model.block_size;
    GGML_ASSERT(block_size > 0);

    const int64_t n_blocks = ubatch.n_seqs_unq;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const size_t token_stride = (size_t) block_drafts * inp_tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, inp_tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    // confidence head input: predicts per-position acceptance
    ggml_tensor * conf_inp = res->t_embd; // [n_embd, n_tok]

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = 0; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);   // [R, n_blocks]
        ggml_tensor * bias    = ggml_mul_mat(ctx0, w2, w1_prev); // [n_vocab, n_blocks]

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
        ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
        ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
        ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
        if (model.dspark_conf_proj_b) {
            conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
        }
        conf = ggml_sigmoid(ctx0, conf);

        cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `llama_get_embeddings_nextn`
        conf = ggml_repeat(ctx0, conf, res->t_embd);
        res->t_h_nextn = conf;
        ggml_build_forward_expand(gf, conf);
    }

    res->t_logits = out;
    ggml_build_forward_expand(gf, out);
}
