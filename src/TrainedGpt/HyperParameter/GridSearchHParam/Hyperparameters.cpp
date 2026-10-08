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
using HValue  = std::variant<int, float>;
using HGrid   = std::vector<std::pair<std::string, std::vector<HValue>>>; // vector keeps order, like a Python dict
using HConfig = std::map<std::string, HValue>;

HGrid makeGrid() {
    return {
            {"batch_size",          {2, 4, 8, 16}},
            {"drop_rate",           {0.0f, 0.1f, 0.2f}},
            {"warmup_iterations",   {10, 20, 30}},
            {"weight_decay",        {0.1f, 0.01f, 0.0f}},
            {"peak_lr",             {0.0001f, 0.0005f, 0.001f, 0.005f}},
            {"initial_lr",          {0.00005f, 0.0001f}},
            {"min_lr",              {0.00005f, 0.00001f, 0.0001f}},
            {"num_epochs",          {5, 10, 15, 20, 25}},
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
    EntropyData entropy_data;
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
               std::vector<torch::Tensor> xs, ys;
               xs.reserve(batch.size());
               ys.reserve(batch.size());
               for (auto &ex : batch) {
                   xs.push_back(ex.data);
                   ys.push_back(ex.target);
               }
               auto inputBatch  = torch::stack(xs).to(device);
               auto targetBatch = torch::stack(ys).to(device);
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
               auto loss = trainer.cal_loss_batch(inputBatch,targetBatch, model);
               loss.backward();

               if (globalStep<=warmup_iterations) {
                   torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
               }
               optimizer.step();
           }
       }
        trainer.evaluate_model(model,trainLoader,valLoader,device,eval_iter);
        return trainer.getEntropyData();
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
    float trainRatio  = 0.75;
    int splitIndex = static_cast<int>(trainRatio * trainer.getText().size());
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");

    torch::manual_seed(123);
    config cfg;
    auto combos = cartesianProduct(makeGrid());
    int currentConfig = 0;
    for (auto &combo : combos)
    {
        try
        {

            currentConfig += 1;
           // std::cout << "currentConfig: " << currentConfig << "\t";
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

            std::cout<<"training starting"<<std::endl;
            gpt2mhl model(cfg);
            model->to(device);
            auto optimizer = torch::optim::AdamW(model->parameters(),
                torch::optim::AdamWOptions(
                std::get<float>(combo.at("peak_lr")))
                .weight_decay(
                    std::get<float>(combo.at("weight_decay"))
                    ));
            std::string startContext = "Nevertheless";
           auto results = trainer_hyperparameter.train_model(model,
                trainloader,
                valLoader,
                optimizer,
                device,
                std::get<int>(combo.at("num_epochs")),
                std::get<int>(combo.at("num_epochs")),
                std::get<int>(combo.at("warmup_iterations"))
                );




            float trainLoss = results.train_losses[-1].item<float>();
            float valLoss   = results.val_targets[-1].item<float>();

            if (valLoss < bestValLoss) {
                bestValLoss     = valLoss;
                bestTrainLoss   = trainLoss;
                bestHParams     = combo;           // copies the whole map
                bestConfigIndex = currentConfig;
            }

            for (const auto &[name, value] : bestHParams) {
                std::cout << "  " << name << " = ";
                std::visit([](auto v) { std::cout << v; }, value);
                std::cout << "\n";
            }
            std::cout<<"training finished"<<std::endl;

        }catch(...)
        {

        }
    }
    std::cout << "Best config #" << bestConfigIndex
<< "  val_loss=" << bestValLoss
<< "  train_loss=" << bestTrainLoss << "\n";


}

