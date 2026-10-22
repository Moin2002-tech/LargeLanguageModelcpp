//
// Created by moinshaikh on 10/6/26.
//

#include<Gpt2Untrained/dataPreparation.hpp>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
#include<doctest.hpp>
#include<iostream>
#include<string>
#include<TrainedGpt/TrainingGuttenbergDatasets/TrainingOnGuttenBerg.hpp>
#include<array>
#include<cmath>
#include<GPT2LargeLanguageModel/util.hpp>
#include<variant>
#include <limits>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAAllocatorConfig.h>

using HValue  = std::variant<int, float>;
using HGrid   = std::vector<std::pair<std::string, std::vector<HValue>>>; // vector keeps order, like a Python dict
using HConfig = std::map<std::string, HValue>;

HGrid makeGrid() {
    // Trimmed grid: 32 configs instead of 12,960.
    // Rationale for this tiny corpus (~14 sliding windows of 256 tokens):
    //  - num_epochs pinned to 5: with this little data the extra epochs mostly
    //    add overfitting noise; retrain the winning config with more epochs.
    //  - warmup/initial_lr/min_lr pinned: with only ~35 total optimizer steps
    //    the schedule shape matters far less than peak_lr itself.
    //  - batch_size 8/16 dropped: they yield only 1-2 batches per epoch here,
    //    i.e. 5-10 update steps total, too few to learn anything.
    //  - drop_rate/weight_decay reduced to the two most common GPT-2 values.
    // Re-expand a dimension here after the first pass if its effect matters.
    return {
            {"batch_size",          {2, 4}},
            {"drop_rate",           {0.0f, 0.1f}},
            {"warmup_iterations",   {10}},
            {"weight_decay",        {0.01f, 0.1f}},
            {"peak_lr",             {0.0001f, 0.0005f, 0.001f, 0.005f}},
            {"initial_lr",          {0.00005f}},
            {"min_lr",              {0.00005f}},
            {"num_epochs",          {5}},
        };
}
std::vector<HConfig> cartesianProduct(const HGrid& grid) {
    std::vector<HConfig> result{HConfig{}};            // start with one empty config
    for (const auto& [name, values] : grid) {
        std::vector<HConfig> next;
        next.reserve(result.size() * values.size());
        for (const auto& partial : result)
            for (const auto& v : values) {
                HConfig c = partial;
                c[name] = v;
                next.push_back(std::move(c));
            }
        result = std::move(next);
    }
    return result;
}

class Hyperparameter
{

    TrainingOnGuttenBerg trainer;

public:
    explicit Hyperparameter( TrainingOnGuttenBerg   &trainer) : trainer{trainer}
    {
    }
    auto train_model(gpt2mhl &model,
        auto &trainLoader,
        auto &valLoader,
        torch::optim::Optimizer &optimizer,
        torch::Device &device,
        int n_epochs,
        int eval_iter,
        int warmup_iterations,
        float initialLearningRate = 3e-05,
        float minRate = 0.1
        )
    {
    int globalStep = 0;
        float maxLearningRate = static_cast<float>(optimizer.param_groups().front().options().get_lr());
        int batchesPerEpoch = 0;
        for (auto &b : *trainLoader) { (void)b; ++batchesPerEpoch; }
        const int totalSteps = batchesPerEpoch * n_epochs;
        // Calculate the learning rate increment at each step during warmup
        float lr_increment = (maxLearningRate - initialLearningRate) / warmup_iterations;
        float progress = 0.0f;
        float lr = 0.0f;
       for (int i = 0; i < n_epochs; ++i) {
           model->train();
           for (auto &batch : *trainLoader)
               {
               const int n = static_cast<int>(batch.size());
               // Forward/backward slice cap. On the 3.62 GiB T600 the static
               // footprint (model + grads + AdamW ~ 2 GiB) plus the autograd
               // graph of a 2 x 256 x 50257 logits pass only just fits:
               // batch_size=2 configs squeaked by with ~0.15 GiB spare, while
               // the same 100 MiB pass OOM'd for effective batch_size=4.
               // Batches larger than 2 therefore run ONE row per pass — the
               // logits allocation halves (100 MiB -> 50 MiB) and the whole
               // activation peak drops ~0.4 GiB. Gradient accumulation still
               // trains at the grid's effective batch size.
               const int microBatchSize = (n <= 2) ? n : 1;
               optimizer.zero_grad();
               globalStep += 1;
               if (globalStep <= warmup_iterations) {
                   lr = initialLearningRate + globalStep * lr_increment;
               }
               else {
                   progress = float(globalStep - warmup_iterations) / float(totalSteps - warmup_iterations);
                   lr = minRate + (maxLearningRate - minRate) * 0.5f * (1.0f + std::cos(float(M_PI) * progress));
               }

               for (auto & param :optimizer.param_groups()) {
                   param.options().set_lr(lr);
               }
               // Forward/backward per micro-batch; gradients accumulate across
               // micro-batches. Each micro loss is scaled by size_micro/n so
               // the accumulated gradient equals the mean over the whole batch.
               for (int c = 0; c < n; c += microBatchSize) {
                   const int hi = std::min(n, c + microBatchSize);
                   std::vector<torch::Tensor> xs, ys;
                   xs.reserve(hi - c);
                   ys.reserve(hi - c);
                   for (int e = c; e < hi; ++e) {
                       xs.push_back(batch[e].data);
                       ys.push_back(batch[e].target);
                   }
                   auto inputBatch  = torch::stack(xs).to(device);
                   auto targetBatch = torch::stack(ys).to(device);
                   auto loss = trainer.cal_loss_batch(inputBatch, targetBatch, model);
                   (loss * static_cast<float>(hi - c) / static_cast<float>(n)).backward();
               }

               if (globalStep<=warmup_iterations) {
                   torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
               }
               optimizer.step();
           }
       }
        // Return the final losses directly. entropy_ is only filled by
        // train_model_simple(), so getEntropyData() would be empty here and
        // indexing train_losses[-1] would throw out-of-range at the call site.
        return trainer.evaluate_model(model,trainLoader,valLoader,device,eval_iter);
    }



};




TEST_CASE("HParamTraining")
{
    std::string verdict = std::string(DATASETS_DIR) + "the-verdict.txt";
    float bestValLoss   = std::numeric_limits<float>::infinity();
    float bestTrainLoss = std::numeric_limits<float>::infinity();
    HConfig bestHParams;                       // same type as your combos
    int bestConfigIndex = -1;
    TrainingOnGuttenBerg trainer(verdict);

    Hyperparameter trainer_hyperparameter(trainer);

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU ;
    // Small-VRAM GPU: expandable segments avoid the "reserved but unallocated"
    // fragmentation dead-ends (same as PYTORCH_CUDA_ALLOC_CONF=
    // expandable_segments:True). Must be set before the first CUDA allocation.
    if (device.is_cuda()) {
        c10::cuda::CUDACachingAllocator::setAllocatorSettings("expandable_segments:True");
    }
    float trainRatio  = 0.75;
    int splitIndex = static_cast<int>(trainRatio * trainer.getText().size());
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");

    torch::manual_seed(123);
    config cfg;
    auto combos = cartesianProduct(makeGrid());
    std::cout << "grid search: " << combos.size() << " total configs" << std::endl;
    int currentConfig = 0;
    for (auto &combo : combos)
    {
        try
        {

            currentConfig += 1;
            std::cout << "config " << currentConfig << "/" << combos.size()
                      << "  num_epochs=" << std::get<int>(combo.at("num_epochs"))
                      << std::endl;
            cfg.vocab_size  = 50257;
            cfg.context_length = 256;
            cfg.emb_dim = 768;
            cfg.drop_rate = std::get<float>(combo.at("drop_rate"));
            cfg.n_heads = 12;
            cfg.n_layer = 12;
            cfg.qkv_bias = false;


            torch::manual_seed(123);
            auto [trainloader, valLoader] = trainer.create_dataloader(
               data.getTokenizer(),
               trainRatio,
               std::get<int>(combo.at("batch_size")),
               cfg.context_length,
               cfg.context_length);


            gpt2mhl model(cfg);
            model->to(device);
            auto optimizer = torch::optim::AdamW(model->parameters(),
                torch::optim::AdamWOptions(
                std::get<float>(combo.at("peak_lr")))
                .weight_decay(
                    std::get<float>(combo.at("weight_decay"))
                    ));
            std::string startContext = "Nevertheless";
            EvalResult results = trainer_hyperparameter.train_model(model,
                trainloader,
                valLoader,
                optimizer,
                device,
                std::get<int>(combo.at("num_epochs")),
                std::get<int>(combo.at("num_epochs")),
                std::get<int>(combo.at("warmup_iterations")),
                std::get<float>(combo.at("initial_lr")),
                std::get<float>(combo.at("min_lr"))
                );

            float trainLoss = results.trainLoss;
            float valLoss   = results.valLoss;
            std::cout << "config " << currentConfig << "/" << combos.size()
                      << " done: train_loss=" << trainLoss
                      << "  val_loss=" << valLoss << std::endl;

            if (valLoss < bestValLoss) {
                bestValLoss     = valLoss;
                bestTrainLoss   = trainLoss;
                bestHParams     = combo;           // copies the whole map
                bestConfigIndex = currentConfig;

                std::cout << "new best (config #" << currentConfig << "):\n";
                for (const auto &[name, value] : bestHParams) {
                    std::cout << "  " << name << " = ";
                    std::visit([](auto v) { std::cout << v; }, value);
                    std::cout << "\n";
                }
            }

        }
        catch(const std::exception &e)
        {
            std::cerr << "config #" << currentConfig << " failed: "
                      << e.what() << std::endl;
        }
        // Hand the previous config's freed (cached) CUDA blocks back to the
        // driver so every config starts from a clean VRAM slate.
        if (device.is_cuda()) {
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
    }
    std::cout << "Best config #" << bestConfigIndex
<< "  val_loss=" << bestValLoss
<< "  train_loss=" << bestTrainLoss << "\n";


}

