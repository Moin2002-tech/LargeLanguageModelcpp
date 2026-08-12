//
// Created by moinshaikh on 8/5/26.
//

#ifndef LARGELANGUAGEMODELCPP_GELUV3_HPP
#define LARGELANGUAGEMODELCPP_GELUV3_HPP

#include <torch/torch.h>
#include <cmath>

class GELUV3Impl : public torch::nn::Module
{
public:
    GELUV3Impl() { }
    torch::Tensor forward(torch::Tensor x) {
        // Use plain double constants so they auto-broadcast to the tensor's
        // device (torch::tensor would create a CPU tensor and cause a
        // CUDA/CPU mismatch when running on GPU).
        const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
        return 0.5 * x * (1 + torch::tanh(sqrt_2_pi * (x + 0.044715 * torch::pow(x, 3))));
    }
};TORCH_MODULE(GELUV3);

#endif //LARGELANGUAGEMODELCPP_GELUV3_HPP
