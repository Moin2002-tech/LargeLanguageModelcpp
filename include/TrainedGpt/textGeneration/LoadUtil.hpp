//
// Created by moinshaikh on 10/29/26.
//

#ifndef LARGELANGUAGEMODELCPP_LOADUTIL_HPP
#define LARGELANGUAGEMODELCPP_LOADUTIL_HPP
#pragma once
#include <torch/torch.h>

inline void copy_into(torch::Tensor dst, const torch::Tensor& src, const char* what) {
    TORCH_CHECK(dst.sizes() == src.sizes(),
                "Shape mismatch for ", what, ": ", dst.sizes(), " vs ", src.sizes());
    torch::NoGradGuard ng;
    dst.copy_(src);
}
#endif //LARGELANGUAGEMODELCPP_LOADUTIL_HPP
