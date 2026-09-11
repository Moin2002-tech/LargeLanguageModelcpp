//
// Created by moinshaikh on 9/5/26.
//


#include<Gpt2Untrained/dataPreparation.hpp>
#include<torch/torch.h>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
#include<basics/GPTDatasetV1.h>
#include<basics/Text.h>
#include<vector>
#include<limits>
#include<doctest.hpp>
#include<c10/cuda/CUDACachingAllocator.h>
#include<torch/serialize.h>
#include<TrainedGpt/generateText.hpp>
#include <torch/script.h>
TEST_CASE("WeightLoading")
{
   config BASE_CONFIG;
    BASE_CONFIG.vocab_size = 50257;
    BASE_CONFIG.context_length = 1024;
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");
    torch::manual_seed(123);
    gpt2mhl model(BASE_CONFIG);

    torch::jit::script::Module container =
        torch::jit::load(std::string(DATASETS_DIR) + "gpt2-small-124M.torchscript");

    auto params  = model->named_parameters(/*recurse=*/true);
    auto buffers = model->named_buffers(/*recurse=*/true);

    torch::NoGradGuard no_grad;
    size_t copied = 0;
    std::vector<std::string> skipped;

    for (const auto& b : container.named_buffers()) {
        std::string key = b.name;

        // undo the "__" -> "." container-export mapping
        size_t pos = 0;
        while ((pos = key.find("__", pos)) != std::string::npos) {
            key.replace(pos, 2, ".");
            pos += 1;
        }

        // checkpoint naming -> gpt2mhl naming
        auto replace_all = [](std::string s, const std::string& from, const std::string& to) {
            size_t p = 0;
            while ((p = s.find(from, p)) != std::string::npos) {
                s.replace(p, from.size(), to);
                p += to.size();
            }
            return s;
        };
        key = replace_all(key, "trf_blocks.", "trfBlock.");
        key = replace_all(key, "norm1.", "layerNorm1.");
        key = replace_all(key, "norm2.", "layerNorm2.");
        key = replace_all(key, "out_proj.", "outProj.");
        key = replace_all(key, "final_norm.", "finalNorm.");
        key = replace_all(key, "out_head.", "outHead.");

        // no counterpart in an MLA model - skip explicitly rather than
        // falling into a silent no-match
        if (key.find("att.mask") != std::string::npos ||
            key.find("att.W_key.") != std::string::npos ||
            key.find("att.W_value.") != std::string::npos) {
            skipped.push_back(key);
            continue;
        }

        if (auto* p = params.find(key)) {
            if (p->sizes() != b.value.sizes()) {
                std::cerr << "SHAPE MISMATCH " << key << ": model=" << p->sizes()
                          << " ckpt=" << b.value.sizes() << "\n";
                continue;
            }
            p->copy_(b.value);
            ++copied;
        } else if (auto* buf = buffers.find(key)) {
            buf->copy_(b.value);
            ++copied;
        } else {
            std::cerr << "NO MATCH in model for checkpoint key: " << key << "\n";
        }
    }

    std::cout << "Copied " << copied << " / " << params.size() << " parameters.\n";
    std::cout << "Intentionally skipped (no MLA counterpart): " << skipped.size() << "\n";
    for (const auto& s : skipped) std::cout << "  " << s << "\n";

    // expect params.size() - (12 blocks * 2 skipped weight tensors) to copy;
    // W_DKV/W_UK/W_UV stay at random init
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    model->to(device);
    model->eval();

    auto idx = data.textToTokenIds("Every effort moves").to(device);

    // sanity check the raw forward pass before trying to generate
    {
       torch::NoGradGuard ng;
       auto logits = model->forward(idx);  // adjust to your model's actual forward signature
       std::cout << "logits has NaN: " << torch::isnan(logits).any().item<bool>() << "\n";
       std::cout << "logits has Inf: " << torch::isinf(logits).any().item<bool>() << "\n";
       std::cout << "logits min/max: " << logits.min().item<float>()
                  << " / " << logits.max().item<float>() << "\n";
    }

    // print raw ids before attempting to decode

std::cout << std::endl;
    auto tokenIds = generate_with_temperature(model, idx, 30, BASE_CONFIG.context_length, 1, 1.5);
    // print raw ids before attempting to decode
    auto ids_cpu = tokenIds.to(torch::kCPU);
    std::cout << "raw token ids: ";
    for (int64_t i = 0; i < ids_cpu.size(1); ++i) {
        std::cout << ids_cpu[0][i].item<int64_t>() << " ";
    }
    auto tokenIds_cpu = tokenIds.to(torch::kCPU);

    std::cout << std::endl;
    std::cout << data.tokenIdsToText(tokenIds_cpu) << std::endl;
}