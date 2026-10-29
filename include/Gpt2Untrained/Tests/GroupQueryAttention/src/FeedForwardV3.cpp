//
// Created by moinshaikh on 8/5/26.
//
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/FeedForwardV3.hpp>
#include<Gpt2Untrained/Tests/GroupQueryAttention/include/GELUV3.hpp>
#include<TrainedGpt/textGeneration/LoadUtil.hpp>

FeedForwardV3Impl::FeedForwardV3Impl(config &cfg) : cfg(cfg) {
    layers = register_module("layers",torch::nn::Sequential(
        torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim,4 * cfg.emb_dim)),
        GELUV3(),
        torch::nn::Linear(torch::nn::LinearOptions(4* cfg.emb_dim,cfg.emb_dim))
        ));
}

torch::Tensor FeedForwardV3Impl::forward(torch::Tensor x) {
    return layers->forward(x);
}

void FeedForwardV3Impl::load_pretrained(const torch::Tensor& fc_w, const torch::Tensor& fc_b,
                                        const torch::Tensor& proj_w, const torch::Tensor& proj_b) {
    auto fc   = layers->ptr<torch::nn::LinearImpl>(0);
    auto proj = layers->ptr<torch::nn::LinearImpl>(2);   // index 1 is GELU

    copy_into(fc->weight,   fc_w.t().contiguous(),   "ff.fc.w");    // [768,3072] -> [3072,768]
    copy_into(fc->bias,     fc_b,                    "ff.fc.b");
    copy_into(proj->weight, proj_w.t().contiguous(), "ff.proj.w");  // [3072,768] -> [768,3072]
    copy_into(proj->bias,   proj_b,                  "ff.proj.b");
}