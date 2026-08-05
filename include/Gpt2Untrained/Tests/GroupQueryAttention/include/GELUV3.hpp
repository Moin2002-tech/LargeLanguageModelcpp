//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_GELUV3_HPP
#define LARGELANGUAGEMODELCPP_GELUV3_HPP

#include <torch/torch.h>

class GELUV3Impl : public torch::nn::Module
{
public:
    GELUV3Impl() { }
    torch::Tensor forward(torch::Tensor x) {
        return 0.5 * x * (1 + torch::tanh(torch::sqrt(torch::tensor({2.0 / M_PI})) *  (x + 0.044715 * torch::pow(x,3)) ));
    }
};TORCH_MODULE(GELUV3);

#endif //LARGELANGUAGEMODELCPP_GELUV3_HPP
