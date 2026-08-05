//
// Created by moinshaikh on 8/5/26.
//


#include<Gpt2Untrained/Tests/GroupQueryAttention/include/Gpt2ModelV3.hpp>


Gpt2ModelV3Impl::Gpt2ModelV3Impl(config &cfg) : cfg(cfg)
{
    tok_emb = register_module("tok_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size,cfg.emb_dim)));
    pos_emb = register_module("pos_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.context_length,cfg.emb_dim)));
    dro_emb = register_module("dro_emb",torch::nn::Dropout(cfg.drop_rate));
    trf_block = register_module("trfBlock", [this, &cfg]()
        {
            torch::nn::ModuleList seq;
            for (int i = 0; i < cfg.n_layer; ++i) {
                seq->push_back(TransformerBlockV4(cfg));
            }
            return seq;
        }());   // <-- () calls the lambda immediately

    finalNorm = register_module("finalNorm", LayerNormV3(cfg.emb_dim));
    outhead = register_module("outhead",torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,cfg.vocab_size).bias(false)))
    ;
}

torch::Tensor Gpt2ModelV3Impl::forward(torch::Tensor x, bool use_Cache)
{
    auto seq_length = x.size(1);
    auto batch_size = x.size(0);

    auto tok_embeds = tok_emb(x);
    torch::Tensor pos_ids;
    //kv_caching
    if (use_Cache) {
        pos_ids = torch::arange(current_pos,current_pos + seq_length,torch::TensorOptions(x.device()).dtype(torch::kLong));
        current_pos += seq_length;
    }
    else
    {
        pos_ids = torch::arange(0,seq_length,torch::TensorOptions(x.device()).dtype(torch::kLong));
    }
    auto pos_embeds = pos_emb(pos_ids).unsqueeze(0);
    x = tok_embeds + pos_embeds;
    x = dro_emb(x);
    for (int i = 0; i < static_cast<int>(trf_block->size()); ++i) {
        auto blk = trf_block->ptr<TransformerBlockV4Impl>(i);
        x = blk->forward(x, use_Cache);
    }

    x= finalNorm(x);
    auto logits = outhead(x);
    return logits;
}

void Gpt2ModelV3Impl::reset_kv_cache()
{

    for (int i = 0; i < static_cast<int>(trf_block->size()); ++i) {
        auto blk = trf_block->ptr<TransformerBlockV4Impl>(i);
        blk->reset_cache();
    }
    // self.ptr_current_pos = 0
    current_pos = 0;
}