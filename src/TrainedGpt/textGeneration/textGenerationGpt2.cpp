//
// Created by moinshaikh on 10/29/26.
//
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
#include<Gpt2Untrained/dataPreparation.hpp>
#include<TrainedGpt/textGeneration/LoadUtil.hpp>


#include<doctest.hpp>

#include "TrainedGpt/textGeneration/ModelLoaderFactory.h"
#include<Gpt2Untrained/dataPreparation.hpp>
#include<TrainedGpt/generateText.hpp>
TEST_CASE("textGeneration") {
    torch::Device device = torch::kCPU;
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");
    auto params = gpt2::load_params(std::string(DATASETS_DIR) + "gpt2/gpt2_124M.torchscript");
    config cfg;
    cfg.qkv_bias = true;
    cfg.drop_rate = 0.0f;
    cfg.latent_dim = 768;
    gpt2mhl model(cfg);
    model->load_pretrained(params);
    model->eval();
    torch::NoGradGuard ng;

    torch::manual_seed(123);
    torch::Tensor idx = data.textToTokenIds("every effort moves you").to(device);
    auto generateTokes = generate_with_temperature(model,idx,700,cfg.context_length,25,0.8);
    std::string text = data.tokenIdsToText(generateTokes);
    std::cout<<text<<std::endl;

}
