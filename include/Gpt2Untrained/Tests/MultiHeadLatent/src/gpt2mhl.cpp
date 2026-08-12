//
// Created by moinshaikh on 8/10/26.
//
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>


gpt2mhlImpl::gpt2mhlImpl(config &cfg) : cfg(cfg)
{
    tok_emb = register_module("tok_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size,cfg.emb_dim)));
    pos_emb = register_module("pos_emb", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.context_length,cfg.emb_dim)));
    drop_emb = register_module("dro_emb", torch::nn::Dropout(cfg.drop_rate));
    trf_block = register_module("trfBlock", [this, &cfg]()
       {
           torch::nn::ModuleList seq;
           for (int i = 0; i < cfg.n_layer; ++i) {
               seq->push_back(TransformBlockV5(cfg));
           }
           return seq;
       }());   // <-- () calls the lambda immediately

    finalNorm = register_module("finalNorm", LayerNormV3(cfg.emb_dim));
    outHead = register_module("outHead",torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,cfg.vocab_size).bias(false)))
    ;

}

torch::Tensor gpt2mhlImpl::forward(torch::Tensor x, bool use_Cache) {
    auto seq_length = x.size(1);
    auto batch_size = x.size(0);
    auto tok_embeds = tok_emb(x);
    torch::Device device = x.device();
    torch::Tensor pos_ids;
    if (use_Cache) {
        pos_ids = torch::arange(current_pos,current_pos+ seq_length,torch::TensorOptions(device).dtype(torch::kLong));
        current_pos += seq_length;
    }
    else {
        pos_ids = torch::arange(0,seq_length,torch::TensorOptions(device).dtype(torch::kLong));
    }
    auto pos_embeds = pos_emb(pos_ids).unsqueeze(0);

    x = tok_embeds + pos_embeds;
    x =  drop_emb(x);
    for (int i = 0; i < static_cast<int>(trf_block->size()); ++i) {
        auto blk = trf_block->ptr<TransformBlockV5Impl>(i);
        x = blk->forward(x, use_Cache);
    }
    x = finalNorm(x);
    auto logits = outHead(x);

    return logits;
}

