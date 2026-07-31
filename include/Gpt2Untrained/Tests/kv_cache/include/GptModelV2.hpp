//
// Created by moinshaikh on 8/1/26.
//

#ifndef LARGELANGUAGEMODELCPP_GPTMODELV2_HPP
#define LARGELANGUAGEMODELCPP_GPTMODELV2_HPP

#include<torch/torch.h>
#include<GPT2LargeLanguageModel/util.hpp>

class GptModelV2Impl : public torch::nn::Module
{
private:
    config cfg;
    torch::nn::Embedding tokembed{nullptr}, postembed{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear outHead{nullptr};
    torch::nn::ModuleList trfBlock{nullptr};
    torch::nn::LayerNorm layernorm{nullptr};
    int current_ptr = 0;
    int kv_window_size = cfg.kv_window_size;
public:
    GptModelV2Impl(config &config);
    torch::Tensor forward(torch::Tensor x, bool use_cache);
    void reset_kv_cache();

    // Accessors needed by the cached text-generation routine
    int64_t getContextLength() const { return postembed->options.num_embeddings(); }
    int64_t getKvWindowSize() const { return kv_window_size; }

};
TORCH_MODULE(GptModelV2);
#endif //LARGELANGUAGEMODELCPP_GPTMODELV2_HPP
