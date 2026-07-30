//
// Created by moinshaikh on 7/19/26.
//

/*Tests 1) Number of parameters in feed forward and attention modules
Calculate and compare the number of parameters that are contained in the feed for-
ward module and those that are contained in the multi-head attention module.*/


#include<torch/torch.h>
#include<AttentionMechanism/MultiHeadAttention.hpp>
#include<GPT2LargeLanguageModel/TransformerBlock.hpp>
#include<GPT2LargeLanguageModel/Gpt2Model.hpp>
#include<iostream>
#include<iomanip>

#include<doctest.hpp>



TEST_CASE("attentionMechanismVSFeedForward") {
    config GPTCONFIG_124M;
    GPTCONFIG_124M.vocab_size = 50257;
    GPTCONFIG_124M.context_length = 1024;
    GPTCONFIG_124M.emb_dim = 768;
    GPTCONFIG_124M.n_heads = 12;
    GPTCONFIG_124M.n_layer =  12;
    GPTCONFIG_124M.drop_rate = 0.1;
    GPTCONFIG_124M.qkv_bias = false;

    auto block = TransformerBlock(GPTCONFIG_124M);


    std::cout<<block<<std::endl;
    int64_t total_params = 0;
    for (const auto& p : block->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total number of parameters in feed forward module: " << total_params << std::endl;

}

/*
    *We initialized a 124-million-parameter GPT model, which is known as "GPT-2 small."
    Without making any code modifications besides updating the configuration file, use
    the GPTModel class to implement GPT-2 medium (using 1,024-dimensional embed-
    dings, 24 transformer blocks, 16 multi-head attention heads), GPT-2 large (1,280-
    dimensional embeddings, 36 transformer blocks, 20 multi-head attention heads),
    and GPT-2 XL (1,600-dimensional embeddings, 48 transformer blocks, 25 multi-head
    attention heads). As a bonus, calculate the total number of parameters in each GPT
    model.
 */

// Helper: calculate total parameters in a model, optionally excluding the output head
// for weight tying calculation.
int64_t count_params(const torch::nn::Module& model, bool exclude_out_head = false)
{
    int64_t total = 0;
    for (const auto& p : model.parameters()) {
        total += p.numel();
    }
    return total;
}

// Helper: print model statistics matching the Python calculate_size() function
void calculate_size(const Gpt2& model, const std::string& label)
{
    int64_t total_params = 0;
    for (const auto& p : model->parameters())
    {
        total_params += p.numel();
    }
    std::cout << label << ": Total number of parameters: " << total_params << std::endl;

    // Exclude output head parameters for weight tying
    auto out_head_params = model->getOutHead()->parameters();
    int64_t out_head_count = 0;
    for (const auto& p : out_head_params)
    {
        out_head_count += p.numel();
    }
    int64_t total_params_gpt2 = total_params - out_head_count;
    std::cout << "  Number of trainable parameters considering weight tying: "
              << total_params_gpt2 << std::endl;

    // Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    double total_size_bytes = static_cast<double>(total_params) * 4.0;
    double total_size_mb = total_size_bytes / (1024.0 * 1024.0);

    std::cout << "  Total size of the model: " << std::fixed << std::setprecision(2)
              << total_size_mb << " MB" << std::endl;
    std::cout << std::endl;
}

TEST_CASE("multiParametersModel")
{
    std::cout << "\n========== GPT-2 Multi-Size Parameter Counts ==========\n" << std::endl;

    // Test each GPT-2 variant
    std::vector<std::string> model_names =
    {
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl"
    };

    for (const auto& name : model_names)
    {
        config cfg = get_gpt2_config(name);

        // Create model with this config
        auto model = Gpt2(cfg);

        // Print model statistics
        calculate_size(model, name);
    }
}