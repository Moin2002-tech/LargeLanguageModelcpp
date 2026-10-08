//
// Created by moinshaikh on 9/24/26.
//

#ifndef LARGELANGUAGEMODELCPP_TRAININGONGUTTENBERG_HPP
#define LARGELANGUAGEMODELCPP_TRAININGONGUTTENBERG_HPP

#include<basics/Text.h>
#include<torch/torch.h>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
#include<tiktoken.hpp>
#include<Gpt2Untrained/dataPreparation.hpp>
#include<basics/GPTDatasetV1.h>
//Initialize lists to track losses and tokens seen
struct EntropyData {
    torch::Tensor train_losses, val_targets,track_token_seen;
    EntropyData()
        : train_losses(torch::empty({0})),
          val_targets(torch::empty({0})),
          track_token_seen(torch::empty({0})) {}
};

struct EvalResult {
    float trainLoss {0.0};
    float valLoss{0.0};
};

class TrainingOnGuttenBerg
{
private:
    std::string text_= {""};
    EntropyData entropy_{};
    EvalResult evalResult_{};

public:
    explicit TrainingOnGuttenBerg(std::string_view path);
   inline auto create_dataloader(std::shared_ptr<tiktoken::Encoding> tokenizer,float training_ratio = 0.0,int batchSize= 0, int maxLength = 0, int stride =0)
    {
       // Character-based split (matches the Python `text[:idx]` / `text[idx:]`
       // reference). training_ratio is now actually used.
       size_t split_idx = static_cast<size_t>(text_.size() * training_ratio);
       std::string train_text = text_.substr(0, split_idx);
       std::string val_text   = text_.substr(split_idx);

       auto train_dataset = GPTDatasetV1(train_text, tokenizer, maxLength, stride);
       auto val_dataset   = GPTDatasetV1(val_text, tokenizer, maxLength, stride);

       auto train_loader =
           torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
               std::move(train_dataset),
               torch::data::DataLoaderOptions().batch_size(batchSize));

       auto val_loader =
           torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
               std::move(val_dataset),
               torch::data::DataLoaderOptions().batch_size(batchSize));

       // NOTE: return type changed from a single loader to a pair.
       return std::make_pair(std::move(train_loader), std::move(val_loader));
   }
    EntropyData train_model_simple(
        gpt2mhl &model,
        torch::optim::Optimizer &optimizer,
        torch::Device &device,
        int n_epochs,
        float evalFreq,
        int printSamplesIteration,
        std::string &start_context,
        std::string &outputDirectory,
        int global_ckpt_freq,
        std::shared_ptr<tiktoken::Encoding> embeddings,
        int batch_size = 1024,
        float trainRatio = 0.70,
        int max_seq_len = 1024);

    torch::Tensor cal_loss_batch(
    const torch::Tensor &inputBatch,
    const torch::Tensor &targetBatch,
    gpt2mhl &model
    );
    template <typename LoaderPtr>
    float total_loss_loader(LoaderPtr &dataLoader, gpt2mhl &model, int numBatches = 0);

    void generateAndPrintSample(
    gpt2mhl& model,
    PreparedData& data,
    torch::Device device,
    const std::string& start_context);

    torch::Tensor generateTextSimpleCachedV2(
    gpt2mhl& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size = 0,
    bool use_cache = true);

    EvalResult evaluate_model(gpt2mhl &model, auto &trainLoader,auto &valLoader,torch::Device device,int iteration)
    {
        model->eval();
        torch::NoGradGuard no_grad;
        auto trainLoss = total_loss_loader(trainLoader,model,iteration);
        auto valLoss = total_loss_loader(valLoader,model,iteration);
        model->train();
        return {trainLoss, valLoss};
    }

    std::string getText() {
        return text_;
    }

    auto getEntropyData() {
        return entropy_;
    }
    auto getEvalResult() {
        return evalResult_;
    }

};


#endif //LARGELANGUAGEMODELCPP_TRAININGONGUTTENBERG_HPP
