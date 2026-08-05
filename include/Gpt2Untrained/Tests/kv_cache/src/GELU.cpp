//
// Created by moinshaikh on 7/31/26.
//
#include<Gpt2Untrained/Tests/kv_cache/include/GELU.hpp>


torch::Tensor GELUImpl::forward(torch::Tensor x)
{
    return 0.5 * x * (1 + torch::tanh(torch::sqrt(torch::tensor({2.0 / M_PI})) *  (x + 0.044715 * torch::pow(x,3)) ));
}
