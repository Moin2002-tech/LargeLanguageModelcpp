//
// Created by moinshaikh on 7/9/26.
//
#include <GPT2LargeLanguageModel/Gpt2Model.hpp>
#include<GPT2LargeLanguageModel/TransformerBlock.hpp>

Gpt2Impl::Gpt2Impl(config& cfg) : cfg(cfg)
{

    tokembed = register_module("tokembed",torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.vocab_size, cfg.emb_dim)));
    postembed = register_module("postembed", torch::nn::Embedding(torch::nn::EmbeddingOptions(cfg.context_length,cfg.emb_dim)));
    dropout = register_module("dropout", torch::nn::Dropout(cfg.drop_rate));
    std::vector<int64_t> lnShape = {static_cast<int64_t>(cfg.emb_dim)};
    layernorm  = register_module("layernorm",  torch::nn::LayerNorm(torch::nn::LayerNormOptions(lnShape)));
    outHead = register_module("outHead",torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(cfg.qkv_bias)));

    //just show off lamda function you can apply differently
    trfBlock = register_module("trfBlock", [this, &cfg]()
        {
            torch::nn::Sequential seq;
            for (int i = 0; i < cfg.n_layer; ++i) {
                seq->push_back(TransformerBlock(cfg));
            }
            return seq;
        }());   // <-- () calls the lambda immediately

}

torch::Tensor Gpt2Impl::forward(torch::Tensor x)
{
    auto batch_size = x.size(0);
    auto seq_length = x.size(1);

    // Token + positional embeddings
    auto tokembeds = tokembed->forward(x);
    auto postembeds = postembed->forward(torch::arange(seq_length, x.device()).to(torch::kLong));
    x = tokembeds + postembeds;

    // Through the layers
    x = dropout->forward(x);
    x = trfBlock->forward(x);
    x = layernorm->forward(x);

    // Output projection
    return outHead->forward(x);
}