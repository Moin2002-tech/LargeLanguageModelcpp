//
// Created by moinshaikh on 7/31/26.
//

#ifndef LARGELANGUAGEMODELCPP_GELU_HPP
#define LARGELANGUAGEMODELCPP_GELU_HPP

#include<torch/torch.h>
#include<torch/nn.h>

class GELUImpl : public torch::nn::Module
{
public:
    GELUImpl() { }
    torch::Tensor forward(torch::Tensor x);

};TORCH_MODULE(GELU);

#endif //LARGELANGUAGEMODELCPP_GELU_HPP
