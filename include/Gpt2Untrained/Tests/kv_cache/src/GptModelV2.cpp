//
// Created by moinshaikh on 8/1/26.
//

#include<Gpt2Untrained/Tests/kv_cache/include/GptModelV2.hpp>
#include<Gpt2Untrained/Tests/kv_cache/include/TransformBlockV2.hpp>

GptModelV2Impl::GptModelV2Impl(config& cfg) : cfg(cfg)
{
    tokembed = register_module("tokembed",torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.emb_dim)));
    postembed = register_module("postembed", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.context_length,cfg.emb_dim)));
    dropout = register_module("dropout", torch::nn::Dropout(cfg.drop_rate));
    std::vector<int64_t> lnShape = {static_cast<int64_t>(cfg.emb_dim)};
    layernorm  = register_module("layernorm",  torch::nn::LayerNorm(torch::nn::LayerNormOptions(lnShape)));
    outHead = register_module("outHead",torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(cfg.qkv_bias)));

    trfBlock = register_module("trfBlock", [this, &cfg]()
        {
            torch::nn::ModuleList seq;
            for (int i = 0; i < cfg.n_layer; ++i) {
                seq->push_back(TransformBlockV2(cfg));
            }
            return seq;
        }());   // <-- () calls the lambda immediately

}

torch::Tensor GptModelV2Impl::forward(torch::Tensor x, bool use_cache) {
    auto seq_length = x.size(1);
    auto batch_size = x.size(0);
    auto tok_embed = tokembed->forward(x);
    torch::Tensor pos_id;
    if (use_cache)
    {
        auto context_length = postembed->options.num_embeddings();
        if (current_ptr + seq_length > context_length) {
            throw std::invalid_argument("Position embedding overflow.");
        }
        pos_id = torch::arange(current_ptr,seq_length + current_ptr,x.options().dtype(torch::kLong).device());
        current_ptr += seq_length;
    }
    else
    {
        pos_id = torch::arange(0,seq_length,x.options().dtype(torch::kLong).device());
    }
    auto pos_embed = postembed->forward(pos_id).unsqueeze(0);

    x = tok_embed + pos_embed;
    x = dropout->forward(x);

    // Equivalent of:
    //     for blk in self.trf_blocks: x = blk(x, use_cache=use_cache)
    for (int i = 0; i < static_cast<int>(trfBlock->size()); ++i) {
        auto blk = trfBlock->ptr<TransformBlockV2Impl>(i);
        x = blk->forward(x, use_cache);
    }

    // final_norm
    x = layernorm->forward(x);

    // logits = self.out_head(x)
    return outHead->forward(x);
}

void GptModelV2Impl::reset_kv_cache()
{
    // for blk in self.trf_blocks: blk.att.reset_cache()
    for (int i = 0; i < static_cast<int>(trfBlock->size()); ++i) {
        auto blk = trfBlock->ptr<TransformBlockV2Impl>(i);
        blk->reset_cache();
    }
    // self.ptr_current_pos = 0
    current_ptr = 0;
}