//
// Created by moinshaikh on 8/7/26.
//

#ifndef LARGELANGUAGEMODELCPP_MULTIHEADLATENTV1_HPP
#define LARGELANGUAGEMODELCPP_MULTIHEADLATENTV1_HPP

#include<torch/torch.h>
#include<torch/nn.h>


class MultiHeadLatentV1Impl : public torch::nn::Module {
private:
    int d_in;
    int d_out;
    int numHeads;
    int headDims;
    int latentDims;
    bool qkv_bias;
    float dropRate;
    int currentPos = 0;

    torch::nn::Linear W_query{nullptr}; //per-head Q
    torch::nn::Linear W_DKV{nullptr}; //down to latent
    torch::nn::Linear W_UK{nullptr}; //latent -> per-head K
    torch::nn::Linear W_UV{nullptr}; //latent -> per-head V

    torch::nn::Linear outProj{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::Tensor cache_c_kv;



public:
    MultiHeadLatentV1Impl(int d_in, int d_out,float dropRate, int numHeads, bool qkv_bias=false, int latentDims = 0);
    void reset_cache();
    torch::Tensor forward(torch::Tensor x, bool use_cache=false);
    torch::Tensor reshape_to_heads(torch::Tensor x, int numHeads, int headDims)
    ;

};TORCH_MODULE(MultiHeadLatentV1);


#endif //LARGELANGUAGEMODELCPP_MULTIHEADLATENTV1_HPP
