//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_GPT2MODELV3_HPP
#define LARGELANGUAGEMODELCPP_GPT2MODELV3_HPP


#include<Gpt2Untrained/Tests/GroupQueryAttention/include/TransformerBlockV4.hpp>

class Gpt2ModelV3Impl : public torch::nn::Module
{
private:
    config cfg;
    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::Embedding pos_emb{nullptr};
    torch::nn::Dropout dro_emb{nullptr};
    torch::nn::ModuleList trf_block{nullptr};
    int current_pos = 0;
    LayerNormV3 finalNorm{nullptr};
    torch::nn::Linear outhead{nullptr};

public:

    Gpt2ModelV3Impl(config &cfg);
    torch::Tensor forward(torch::Tensor x,bool use_Cache = false);
    void reset_kv_cache();
    int64_t getContextLength() const { return pos_emb->options.num_embeddings(); }
    int64_t getKvWindowSize() const { return static_cast<int64_t>(cfg.kv_window_size); }

};TORCH_MODULE(Gpt2ModelV3);
#endif //LARGELANGUAGEMODELCPP_GPT2MODELV3_HPP
