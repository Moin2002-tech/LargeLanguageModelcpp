//
// Created by moinshaikh on 9/25/26.
//
// C++ translation of the Python `__main__` training driver for the "pretrain
// on Gutenberg" GPT model. The Python original used argparse; here there is NO
// command-line parser -- every hyper-parameter is a hard-coded variable whose
// defaults match the Python argparse defaults exactly.
//
#include <TrainedGpt/TrainingGuttenbergDatasets/TrainingOnGuttenBerg.hpp>

#include <doctest.hpp>
#include <GPT2LargeLanguageModel/util.hpp>
#include <Gpt2Untrained/dataPreparation.hpp>
#include <Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

TEST_CASE("bergTrainingSession")
{
    // --- 1. Hyper-parameters (Python argparse defaults, no parser) --------
    std::string data_path  = std::string(DATASETS_DIR) + "guttenberg_preprocessed/combined_1.txt";
    std::string output_dir = "model_checkpoints";

    const int    n_epochs          = 1;
    const int    print_sample_iter = 1000;
    const int    eval_freq         = 100;
    const int    save_ckpt_freq    = 100000;
    const double lr                = 5e-4;
    const int    batch_size        = 4;
    const bool   debug             = false;

    // --- 2. Model config (Python GPT_CONFIG_124M dict -> config struct) ---
    config GPT_CONFIG_124M;   // struct defaults == full GPT-2 124M
    if (debug)
    {
        GPT_CONFIG_124M.vocab_size     = 50257;
        GPT_CONFIG_124M.context_length = 1024;
        GPT_CONFIG_124M.emb_dim        = 12;
        GPT_CONFIG_124M.n_heads        = 12;
        GPT_CONFIG_124M.n_layer        = 12;
        GPT_CONFIG_124M.drop_rate      = 0.0f;
        GPT_CONFIG_124M.qkv_bias       = false;
    }

    // --- 3. Device, seed, model, optimizer, tokenizer ----------------------
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::manual_seed(123);

    gpt2mhl model(GPT_CONFIG_124M);
    model->to(device);

    auto optimizer = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(lr).weight_decay(0.1));

    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken"); // gpt2 encoding

    // --- 4. Data: the corpus is already combined into a single text file ----
    // (Your Python version walked a dir of *.txt files; those were pre-merged
    //  into datasets/guttenberg_preprocessed/combined_1.txt, so we use it
    //  directly -- no recursive scan / merge needed.)
    if (!fs::exists(fs::path(data_path)))
    {
        std::cout << "No training text files found. Make sure you "
                     "selected the correct input directory\n";
        return;
    }

    fs::create_directories(output_dir);
    TrainingOnGuttenBerg trainer(data_path);

    // --- 6. Training (mirrors train_model_simple call) ----------------------
    std::string startContext = "Every effort moves you";
    EntropyData entropy = trainer.train_model_simple(
        model, optimizer, device,
        /*n_epochs=*/              n_epochs,
        /*evalFreq=*/              static_cast<float>(eval_freq),
        /*printSamplesIteration=*/ print_sample_iter,
        startContext,
        output_dir,
        /*global_ckpt_freq=*/      save_ckpt_freq,
        data.getTokenizer(),
        /*batch_size=*/            batch_size,
        /*trainRatio=*/            0.7f);

    // --- 7. Loss "plot" (plot_losses replacement: text table) ---------------
    std::cout << "\n=== Loss curve (step | tokens | train | val) ===\n";
    if (entropy.train_losses.size(0) > 0)
    {
        // train_losses/val_targets/track_token_seen can be empty & Float (if no
        // eval step ran) or Int64 (after an eval step), so normalise to float.
        torch::Tensor tLossT = entropy.train_losses.to(torch::kFloat);
        torch::Tensor vLossT = entropy.val_targets.to(torch::kFloat);
        torch::Tensor toksT  = entropy.track_token_seen.to(torch::kFloat);
        auto tLoss = tLossT.accessor<float, 1>();
        auto vLoss = vLossT.accessor<float, 1>();
        auto toks  = toksT.accessor<float, 1>();
        for (int64_t i = 0; i < entropy.train_losses.size(0); ++i)
        {
            std::cout << static_cast<int64_t>(i) << "\t" << toks[i] << "\t"
                      << tLoss[i] << "\t" << vLoss[i] << "\n";
        }
    }

    // --- 8. Save final model ------------------------------------------------
    std::string finalPath = std::string(DATASETS_DIR) + "model_pg_final.torchscript";
    torch::save(model, finalPath);
    std::cout << "Saved final model: " << finalPath << "\n";

    // --- 9. Peak CUDA memory ------------------------------------------------
    if (device.is_cuda())
    {
        try
        {
            int device_id = c10::cuda::current_device();
            c10::cuda::CUDACachingAllocator::DeviceStats stats =
                c10::cuda::CUDACachingAllocator::getDeviceStats(device_id);
            int64_t peakBytes = stats.allocated_bytes[0].peak;
            std::cout << "Maximum GPU memory allocated: "
                      << (static_cast<double>(peakBytes) / 1e9) << " GB\n";
        }
        catch (const std::exception&) { /* diagnostic only */ }
    }
}