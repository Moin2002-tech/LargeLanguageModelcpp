//
// Created by moinshaikh on 9/25/26.
//
#include<TrainedGpt/TrainingGuttenbergDatasets/TrainingOnGuttenBerg.hpp>

#include<TrainedGpt/TrainingGuttenbergDatasets/TrainingOnGuttenBerg.hpp>
#include<basics/GPTDatasetV1.h>
#include<memory>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <atomic>

namespace {
    // SIGINT doesn't throw a C++ exception, so a try/catch(std::exception&)
    // around the training loop (as in the original skeleton) will NOT catch
    // Ctrl+C. We use a flag set from a signal handler and check it each step.
    std::atomic<bool> g_interrupted{false};
    void handle_sigint(int) { g_interrupted = true; }
}
TrainingOnGuttenBerg::TrainingOnGuttenBerg(std::string_view path) {
   Text text(path);
   text_ = text.getText();
}

auto TrainingOnGuttenBerg::create_dataloader(
    std::shared_ptr<tiktoken::Encoding> tokenizer,
    float training_ratio,
    int batchSize,
    int maxLength,
    int stride)
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

torch::Tensor TrainingOnGuttenBerg::cal_loss_batch(const torch::Tensor &inputBatch, const torch::Tensor &targetBatch, gpt2mhl &model)
{
   // Use the model's device so input/target always match the model weights.
   // This avoids "index is on cuda:0, different from other tensors on cpu"
   // when the model hasn't been moved to the expected device.
   auto params = model->parameters();
   torch::Device device = params.empty() ? torch::kCPU : params[0].device();
   torch::Tensor input = inputBatch.to(device);
   torch::Tensor target = targetBatch.to(device);
   // Use use_cache=false: the KV-cache must NOT be used during loss
   // evaluation. With use_cache=true, the model's `current_pos` member
   // accumulates across every batch and is never reset, so after enough
   // batches `pos_ids` exceeds the positional-embedding table size
   // (context_length), causing the CUDA gather index-out-of-bounds.
   torch::Tensor logits = model(input, false);
   torch::Tensor loss = torch::nn::functional::cross_entropy(logits.flatten(0, 1), target.flatten());
   return loss;
}

template<typename LoaderPtr>
float TrainingOnGuttenBerg::total_loss_loader(LoaderPtr &dataLoader, gpt2mhl &model, int numBatches)
{
   // The C++ DataLoader iterator is single-pass (it cannot be reset reliably),
   // so buffer all batches into memory first. Each batch is a
   // std::vector<Example<>>; we stack the examples into a single
   // [batch_size, seq_len] tensor, matching Python's input_batch/target_batch.
   std::vector<torch::Tensor> inputBatches;
   std::vector<torch::Tensor> targetBatches;
   for (auto &batch : *dataLoader) {
      std::vector<torch::Tensor> inputs;
      std::vector<torch::Tensor> targets;
      inputs.reserve(batch.size());
      targets.reserve(batch.size());
      for (const auto &example : batch) {
         inputs.push_back(example.data);
         targets.push_back(example.target);
      }
      inputBatches.push_back(torch::stack(inputs, 0));
      targetBatches.push_back(torch::stack(targets, 0));
   }


   if (inputBatches.empty()) {
      return std::numeric_limits<float>::quiet_NaN();
   }

   // Python: `num_batches is None` -> use all; otherwise clamp with min().
   int numBatchesInLoader = static_cast<int>(inputBatches.size());
   if (numBatches == 0)
   {
      numBatches = numBatchesInLoader;
   } else
   {
      numBatches = std::min(numBatches, numBatchesInLoader);
   }

   float totalLoss = 0.0f;
   for (int i = 0; i < numBatches; ++i) {
      torch::Tensor loss = cal_loss_batch(inputBatches[i], targetBatches[i], model);
      totalLoss += loss.item<float>();
   }


   return totalLoss / numBatches;
}

torch::Tensor TrainingOnGuttenBerg::generateTextSimpleCachedV2(gpt2mhl &model, torch::Tensor idx, int max_new_tokens, int context_size, bool use_cache)
{
   model->eval();

   int64_t ctx_len = (context_size > 0)
       ? static_cast<int64_t>(context_size)
       : model->getContextLength();

   torch::NoGradGuard no_grad;

   if (use_cache) {
      model->reset_kv_cache();

      // Pre-fill with the full prompt (restricted to the last ctx_len tokens).
      auto start_col = std::max(static_cast<int64_t>(0), idx.size(1) - ctx_len);
      torch::Tensor input_tokens = idx.slice(/*dim=*/1, /*start=*/start_col);
      int64_t input_tokens_length = input_tokens.size(1);

      torch::Tensor logits = model->forward(input_tokens, /*use_cache=*/true);

      // Can't generate more than ctx_len total result tokens due to the
      // limitation of the (learnt) position embedding.
      int64_t max_generable = ctx_len - input_tokens_length;
      max_new_tokens = static_cast<int>(std::min<int64_t>(max_new_tokens, max_generable));

      for (int step = 0; step < max_new_tokens; ++step) {
         // Greedy decode: [batch, vocab] -> [batch, 1]
         torch::Tensor next_idx = logits.select(/*dim=*/1, /*index=*/-1)
                                      .argmax(/*dim=*/-1, /*keepdim=*/true);
         idx = torch::cat({idx, next_idx}, /*dim=*/1);

         // Feed only the single new token back; attention uses the KV-cache.
         logits = model->forward(next_idx, /*use_cache=*/true);
      }
   } else {
      for (int step = 0; step < max_new_tokens; ++step) {
         auto start_col = std::max(static_cast<int64_t>(0), idx.size(1) - ctx_len);
         torch::Tensor logits = model->forward(
             idx.slice(/*dim=*/1, /*start=*/start_col),
             /*use_cache=*/false
         );

         torch::Tensor next_idx = logits.select(/*dim=*/1, /*index=*/-1)
                                      .argmax(/*dim=*/-1, /*keepdim=*/true);
         idx = torch::cat({idx, next_idx}, /*dim=*/1);
      }
   }

   return idx;
}

void TrainingOnGuttenBerg::generateAndPrintSample(gpt2mhl &model, PreparedData &data, torch::Device device, const std::string &start_context)
{
   model->eval();

   // Equivalent to Python's model.pos_emb.weight.shape[0]
   int context_size = model->getContextLength();

   // text_to_token_ids(start_context, tokenizer).to(device)
   torch::Tensor encoded = data.textToTokenIds(start_context).to(device);

   // generateTextSimpleCachedV2 already wraps its body in torch::NoGradGuard,
   // so it's the C++ equivalent of the `with torch.no_grad():` block.
   torch::Tensor token_ids = generateTextSimpleCachedV2(
       model, encoded, /*max_new_tokens=*/50, context_size);

   // token_ids_to_text(token_ids, tokenizer)
   std::string decoded_text = data.tokenIdsToText(token_ids);

   // print(decoded_text.replace("\n", " "))  — compact print format.
   // Build the string with newlines replaced by spaces.
   std::string compact;
   compact.reserve(decoded_text.size());
   for (char c : decoded_text) {
      compact += (c == '\n') ? ' ' : c;
   }
   std::cout << compact << std::endl;

   model->train();  // Restore training mode
}


EntropyData TrainingOnGuttenBerg::train_model_simple(gpt2mhl &model,
    torch::optim::Optimizer &optimizer, torch::Device &device,
    int n_epochs,
    float evalFreq,
    int printSamplesIteration,
    std::string &start_context,
    std::string &outputDirectory,
    int global_ckpt_freq,
    std::shared_ptr<tiktoken::Encoding> embeddings,
    int batch_size,
    float trainRatio
    )
{
    // Operate on the class's own members rather than shadowing them, so
    // getEntropyData()/getEvalResult() reflect this run afterwards.
    entropy_ = EntropyData{};
    int token_seen  = 0;
    int global_step = 0;

    int context_length = model->getContextLength();

    // Build the training/validation loaders once, up front, from the single
    // combined Gutenberg text file — no per-book loop needed since it's
    // already merged. stride == context_length matches the Python reference
    // (non-overlapping chunks).
    auto [trainLoader, valLoader] = create_dataloader(
        embeddings, trainRatio, batch_size, context_length, context_length);

    // ASSUMPTION: PreparedData(tokenizer) is a valid constructor — adjust if
    // dataPreparation.hpp declares it differently.
    PreparedData preparedData(embeddings);

    std::filesystem::create_directories(outputDirectory);

    // Install SIGINT handler so Ctrl+C saves a checkpoint instead of just
    // killing the process mid-batch.
    std::signal(SIGINT, handle_sigint);

    auto start_time = std::chrono::steady_clock::now();

    auto save_checkpoint = [&](const std::string &tag) {
        std::string path = outputDirectory + "/model_" + tag + ".pt";
        torch::save(model, path);
        std::cout << "Saved checkpoint: " << path << "\n";
    };

    for (int epoch = 0; epoch < n_epochs && !g_interrupted; ++epoch) {
        model->train();

        for (auto &batch : *trainLoader) {
            if (g_interrupted) break;

            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> targets;
            inputs.reserve(batch.size());
            targets.reserve(batch.size());
            for (const auto &example : batch) {
                inputs.push_back(example.data);
                targets.push_back(example.target);
            }
            torch::Tensor inputBatch  = torch::stack(inputs, 0);
            torch::Tensor targetBatch = torch::stack(targets, 0);

            optimizer.zero_grad();
            auto loss = cal_loss_batch(inputBatch, targetBatch, model);
            loss.backward();
            optimizer.step();
            optimizer.zero_grad(/*set_to_none=*/true);

            token_seen  += static_cast<int>(inputBatch.numel());
            global_step += 1;

            // --- Evaluation ---
            if (global_step % static_cast<int>(evalFreq) == 0) {
                // Fresh eval-only loaders each time — see note above on why
                // trainLoader/valLoader can't be reused here.
                auto [evalTrainLoader, evalValLoader] = create_dataloader(
                    embeddings, trainRatio, batch_size, context_length, context_length);

                // NOTE: train_model_simple has no explicit "eval_iter" param
                // (how many batches to average over). Passing 0 here means
                // total_loss_loader evaluates over ALL batches in the split,
                // which is correct but can be slow on a large corpus. Add an
                // eval_iter parameter if you want to cap it (e.g. 5 batches).
                evalResult_ = evaluate_model(model, evalTrainLoader, evalValLoader, device, /*iteration=*/0);

                auto train_loss_1d  = torch::tensor({evalResult_.trainLoss});
                auto val_loss_1d    = torch::tensor({evalResult_.valLoss});
                auto token_seen_1d  = torch::tensor({static_cast<int64_t>(token_seen)}, torch::kInt64);

                entropy_.train_losses     = torch::cat({entropy_.train_losses, train_loss_1d}, 0);
                entropy_.val_targets      = torch::cat({entropy_.val_targets, val_loss_1d}, 0);
                entropy_.track_token_seen = torch::cat({entropy_.track_token_seen, token_seen_1d}, 0);

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time).count();

                std::cout << "Epoch: " << epoch + 1
                          << "\tStep: " << global_step
                          << "\tTrainLoss: " << evalResult_.trainLoss
                          << "\tValLoss: " << evalResult_.valLoss
                          << "\tElapsed: " << elapsed << "s\n";
            }

            // --- Sample generation ---
            if (printSamplesIteration > 0 && global_step % printSamplesIteration == 0) {
                generateAndPrintSample(model, preparedData, device, start_context);
            }

            // --- Checkpointing ---
            if (global_ckpt_freq > 0 && global_step % global_ckpt_freq == 0) {
                save_checkpoint(std::to_string(global_step));
            }
        }
    }

    if (g_interrupted) {
        std::cout << "Interrupted — saving checkpoint before exit.\n";
        save_checkpoint("interrupted");
    }

    std::signal(SIGINT, SIG_DFL);  // restore default handler
    return entropy_;
}
