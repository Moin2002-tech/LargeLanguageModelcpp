//
// Created by moinshaikh on 8/21/26.
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

auto createDataLoaderV2(const std::string &text,
    std::shared_ptr<tiktoken::Encoding> tokenizer,
    int batch_size,
    int max_length,
    int stride
    )
{
    auto dataset = GPTDatasetV1(text, tokenizer, max_length, stride);
    // Create dataloader
    // SequentialSampler is used to match the default behavior of Python's
    // DataLoader (shuffle=False)
    auto dataloader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size));

    return dataloader;
}

torch::Tensor generateTextSimpleCachedV2(
    gpt2mhl& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size = 0,
    bool use_cache = true)
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

// C++ translation of the Python's generate_and_print_sample():
//   def generate_and_print_sample(model, tokenizer, device, start_context):
//       model.eval()
//       context_size = model.pos_emb.weight.shape[0]
//       encoded = text_to_token_ids(start_context, tokenizer).to(device)
//       with torch.no_grad():
//           token_ids = generate_text_simple(
//               model=model, idx=encoded,
//               max_new_tokens=50, context_size=context_size
//           )
//       decoded_text = token_ids_to_text(token_ids, tokenizer)
//       print(decoded_text.replace("\n", " "))  // Compact print format
//       model.train()
void generateAndPrintSample(
    gpt2mhl& model,
    PreparedData& data,
    torch::Device device,
    const std::string& start_context)
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


/*
* - Next, we implement a utility function to calculate the cross-entropy loss of a given batch
- In addition, we implement a second utility function to compute the loss for a user-specified number of batches in a data loader
 */





torch::Tensor cal_loss_batch(
    const torch::Tensor &inputBatch,
    const torch::Tensor &targetBatch,
    gpt2mhl &model
    )
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


template <typename LoaderPtr>
float total_loss_loader(LoaderPtr &dataLoader, gpt2mhl &model, int numBatches = 0)
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


struct EvalResult {
    float trainLoss;
    float valLoss;
};

EvalResult evaluate_model(gpt2mhl &model, auto &trainLoader,auto &valLoader,torch::Device device,int iteration)
{
    model->eval();
    torch::NoGradGuard no_grad;
    auto trainLoss = total_loss_loader(trainLoader,model,iteration);
    auto valLoss = total_loss_loader(valLoader,model,iteration);
    model->train();
    return {trainLoss, valLoss};
}
//Initialize lists to track losses and tokens seen
struct EntropyData {
    torch::Tensor train_losses, val_targets,track_token_seen;
    EntropyData()
        : train_losses(torch::empty({0})),
          val_targets(torch::empty({0})),
          track_token_seen(torch::empty({0})) {}
};
EntropyData train_model_simple(gpt2mhl &model,PreparedData &preparedData,
    auto &trainLoader,
    auto &valLoader,
    torch::optim::Optimizer &optimizer,
    torch::Device device,
    int numEpochs,
    float evalFrequency,
    int iteration,
    const std::string &start_context,
    std::shared_ptr<tiktoken::Encoding> tokenizer
    )
{
    EntropyData CrosEntropy;
    int token_seen = 0;
    int global_step = -1;

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        model->train();
        for (auto &batch : *trainLoader)
        {
            // The DataLoader iterator yields a std::vector<Example<>> (a single
            // batch). Stack all examples into [batch_size, seq_len] tensors,
            // matching the input/target batch layout expected by cal_loss_batch.
            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> targets;
            inputs.reserve(batch.size());
            targets.reserve(batch.size());
            for (const auto &example : batch) {
                inputs.push_back(example.data);
                targets.push_back(example.target);
            }
            torch::Tensor inputBatch = torch::stack(inputs, 0);
            torch::Tensor targetBatch = torch::stack(targets, 0);

            optimizer.zero_grad();//Reset loss gradients from previous batch iteration
            auto loss = cal_loss_batch(inputBatch, targetBatch, model);
            loss.backward();// Calculate loss gradients
            optimizer.step();//Update model weights using loss gradients
            token_seen += static_cast<int>(inputBatch.numel());
            global_step += 1;

            //Optional evaluation step
            // Evaluation / tracking step
            if (global_step % numEpochs == 0) {
                auto [trainLoss, valLoss] = evaluate_model(model,trainLoader,valLoader,device,iteration);

                // 1. Reshape scalar losses into 1D tensors.
                // Keep these on CPU: EntropyData's tensors are initialized on
                // CPU (torch::empty({0})), so cat() would fail with a device
                // mismatch if we moved them to the GPU device.
                auto train_loss_1d = torch::tensor({trainLoss});
                auto val_loss_1d = torch::tensor({valLoss});
                auto token_seen_1d = torch::tensor({static_cast<int64_t>(token_seen)}, torch::kInt64);

                // 2. Concatenate values onto end of respective tensors
                CrosEntropy.train_losses = torch::cat({CrosEntropy.train_losses, train_loss_1d}, /*dim=*/0);
                CrosEntropy.val_targets = torch::cat({CrosEntropy.val_targets, val_loss_1d}, /*dim=*/0);
                CrosEntropy.track_token_seen = torch::cat({CrosEntropy.track_token_seen, token_seen_1d}, /*dim=*/0);
                std::cout<<"Epochs: "<< epoch+1<<"\t"<<"Step: "<<global_step<<"\t"<<"TrainLoss: "<<trainLoss<<"\t"<<"Val Loss: "<<valLoss<<"\n";
            }

        }
        generateAndPrintSample(model,preparedData,device,start_context);
    }

    return CrosEntropy;
}



TEST_CASE("PreTrainingOnunlabledData")
{
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");
    torch::Tensor batch = data.encodeBatch({"Every effort moves you"});
    config cfg;
    gpt2mhl gpt2Mhl(cfg);

    auto out = gpt2Mhl(batch);
    std::cout << "Initial output shape: " << out.sizes() << std::endl;

    torch::Tensor generatedText = generateTextSimpleCachedV2(gpt2Mhl,batch,10,cfg.context_length);
    std::string decoded_text = data.decode(generatedText.cpu());
    std::cout << "Output text: " << decoded_text << "\n";
    /*
    Testing started at 10:43 PM ...
    Initial output shape: [1, 4, 50257]
    Output text: Every effort moves you monkeyOOKackets superficial)].Estgan Embables possible
    */

    /*Clearly, the model isn’t yet producing coherent text because it hasn’t undergone
    training. To define what makes text “coherent” or “high quality,” we have to imple-
    ment a numerical method to evaluate the generated content. This approach will
    enable us to monitor and enhance the model’s performance throughout its training
    process.
    Next, we will calculate a loss metric for the generated outputs. This loss serves as a
    progress and success indicator of the training progress. Furthermore, in later chap-
    ters, when we fine-tune our LLM, we will review additional methodologies for assess-
    ing model quality.*/


    //- Suppose we have an `inputs` tensor containing the token IDs for 2 training examples (rows)
    //- Corresponding to the `inputs`, the `targets` contain the desired token IDs that we want the model to generate
    // - Notice that the `targets` are the `inputs` shifted by 1 position, as explained in chapter 2 when we implemented the data loader

    torch::Tensor input = torch::tensor({{16833, 3626, 6100},{40,    1107, 588}}); // every effort moves you
    // I really like
    torch::Tensor target = torch::tensor({{3626, 6100, 345 },{1107,  588, 11311}}); // every effort moves you
    //I really like chocolate

    // Scope the NoGradGuard so it ends BEFORE the training loop. Otherwise
    // gradients stay disabled and loss.backward() fails with "does not
    // require grad and does not have a grad_fn".
    {
        torch::NoGradGuard no_grad;
        torch::Tensor logits = gpt2Mhl(input);
        torch::Tensor probabilities = torch::softmax(logits, -1);
        std::cout<<"probabilities: "<<probabilities.sizes()<<std::endl;

        torch::Tensor token_ids = torch::argmax(probabilities,-1,true);
        std::cout<<"token_ids: "<<token_ids.sizes()<<std::endl;

        std::cout<<"Target Batch1: "<< data.tokenIdsToText(target[0].flatten())<<std::endl;
        std::cout<<"Result Batch1"<< data.tokenIdsToText(token_ids[0].flatten())<<std::endl;

        /*
        * Target Batch1:  effort moves you
        Result Batch1 disappointmentaddy whispers
        Process finished with exit code 0
         */
        //- That's because the model wasn't trained yet
        //To train the model, we need to know how far it is away from the correct predictions (targets)

        int text_ids = 0;
        torch::Tensor log_probability1 = probabilities.index({text_ids,torch::tensor({0,1,2}),target.index({text_ids})});
        std::cout<<"target_1: "<<log_probability1.sizes()<<std::endl;
        std::cout<<"target_1: "<<log_probability1<<std::endl;

        int text_ids2 = 1;
        torch::Tensor log_probability2 = probabilities.index({text_ids2,torch::tensor({0,1,2}),target.index({text_ids2})});
        std::cout<<"target_2: "<<log_probability2.sizes()<<std::endl;
        std::cout<<"target_2: "<<log_probability2<<std::endl;

        torch::Tensor log_probs = torch::log(torch::cat({log_probability1,log_probability2}));
        std::cout<<"log_probs: "<<log_probs.sizes()<<std::endl;
        std::cout<<"log_probs: "<<log_probs<<std::endl;

        //average log probability
        torch::Tensor avg_log_probs = torch::mean(log_probs);
        std::cout<<"avg_log_probs: "<< avg_log_probs<<std::endl;


        /*
        * - The goal is to make this average log probability as large as possible by optimizing the model weights
        - Due to the log, the largest possible value is 0, and we are currently far away from 0
         */

        /*
        * - In deep learning, instead of maximizing the average log-probability, it's a standard convention to minimize the *negative* average log-probability value; in our case, instead of maximizing -10.7722 so that it approaches 0, in deep learning, we would minimize 10.7722 so that it approaches 0
        - The value negative of -10.7722, i.e., 10.7722, is also called cross-entropy loss in deep learning
         */

        torch::Tensor neg_log_probs = avg_log_probs * -1;
        std::cout<<"neg_log_probs: "<< neg_log_probs<<std::endl;
    }

    Text text(std::string(DATASETS_DIR) + "the-verdict.txt");
    text.printText(0,99);


    const std::string& rawText = text.getText();
    constexpr double train_ratio = 0.90;
    size_t split_idx = static_cast<size_t>(train_ratio * rawText.size());
    std::string train_data = rawText.substr(0, split_idx);
    std::string val_data = rawText.substr(split_idx);

    std::cout << "Total text length: " << rawText.size() << std::endl;
    std::cout << "Train length: " << train_data.size() << std::endl;
    std::cout << "Val length: " << val_data.size() << std::endl;
    torch::manual_seed(123);

    auto trainLoader = createDataLoaderV2(train_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);
    auto valLoader = createDataLoaderV2(val_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);

    for (auto& batch : *trainLoader) {            // batch is std::vector<Example<>>
        for (auto& example : batch) {            // example is the Example<> class
            auto input = example.data;
            auto target = example.target;

            for (int i = 0; i < 4; i++)
            {
                std::cout << "Input: " << input[i] << std::endl;
                std::cout << "Target: " << target[i] << std::endl;
            }
            std::cout << "Batch info:" << std::endl;
            std::cout << "  Input shape: " << input.sizes() << std::endl;
            std::cout << "  Target shape: " << target.sizes() << std::endl;

            // Print first example in batch to verify sliding window logic (target =
            // input shifted by 1)
            std::cout << "  First input in batch: " << input[0] << std::endl;
            std::cout << "  First target in batch: " << target[0] << std::endl;



        }
    }
    // Calculate tokens using the same logic as the PyTorch version.

    int64_t train_tokens = 0;
    for (auto &input_batch : *trainLoader) {           // each vector<Example<>> is a batch
        for (const auto &example : input_batch) {      // sum over all examples in the batch
            train_tokens += example.data.numel();      // == batch_size * seq_len per batch
        }
    }

    int64_t val_tokens = 0;
    for (auto &input_batch : *valLoader) {
        for (const auto &example : input_batch) {
            val_tokens += example.data.numel();
        }
    }

    std::cout << "Training tokens: " << train_tokens << std::endl;
    std::cout << "Validation tokens: " << val_tokens << std::endl;
    std::cout << "All tokens: " << (train_tokens + val_tokens) << std::endl;

    //- Next, we implement a utility function to calculate the cross-entropy loss of a given batch
    //- In addition, we implement a second utility function to compute the loss for a user-specified number of batches in a data loader


    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU ;


    // IMPORTANT: PyTorch C++ data loaders are single-pass iterators. The
    // trainLoader and valLoader were already consumed by the loops above, so
    // create FRESH loaders here before computing the loss. Otherwise
    // total_loss_loader would see 0 batches and return NaN.
    torch::manual_seed(123);
    auto freshTrainLoader = createDataLoaderV2(train_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);
    auto freshValLoader = createDataLoaderV2(val_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);


        // Match Python's `with torch.no_grad():` when calling calc_loss_loader.
        // IMPORTANT: scope the NoGradGuard so it ends BEFORE train_model_simple
        // is called. Otherwise gradients stay disabled and loss.backward() fails
        // with "does not require grad and does not have a grad_fn".
        float trainloss, valloss;
        {
            torch::NoGradGuard no_grad1;
            trainloss = total_loss_loader(freshTrainLoader, gpt2Mhl);
            valloss = total_loss_loader(freshValLoader, gpt2Mhl);
        }

        std::cout << "Training loss: " << trainloss << std::endl;
        std::cout << "Validation loss: " << valloss << std::endl;

    // `gpt2Mhl` is reused directly for training (no second model is created),
    // so only one full GPT-2-scale model (~124M params) is on the GPU at a
    // time, avoiding CUDA out-of-memory.

    torch::manual_seed(123);
/*
    auto optimizer = torch::optim::AdamW(gpt2Mhl->parameters(),torch::optim::AdamWOptions(0.0004).weight_decay(0.1));
    int num_epochs = 10;
    EntropyData data2 = train_model_simple(gpt2Mhl,
        trainLoader,
        valLoader,
        optimizer,
        device, num_epochs, 5, 5, "every effort moves you",data.getTokenizer());
*/



}


TEST_CASE("TrainingSession")
{
    PreparedData data(std::string(DATASETS_DIR) + "gpt2.tiktoken");
    torch::Tensor batch = data.encodeBatch({"Every effort moves you"});
    config cfg;
    gpt2mhl gpt2Mhl(cfg);

    Text text(std::string(DATASETS_DIR) + "the-verdict.txt");
    text.printText(0,99);


    const std::string& rawText = text.getText();
    constexpr double train_ratio = 0.70;
    size_t split_idx = static_cast<size_t>(train_ratio * rawText.size());
    std::string train_data = rawText.substr(0, split_idx);
    std::string val_data = rawText.substr(split_idx);

    std::cout << "Total text length: " << rawText.size() << std::endl;
    std::cout << "Train length: " << train_data.size() << std::endl;
    std::cout << "Val length: " << val_data.size() << std::endl;
    torch::manual_seed(123);
    torch::Device device = torch::kCPU;
    auto trainLoader = createDataLoaderV2(train_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);
    auto valLoader = createDataLoaderV2(val_data, data.getTokenizer(), 2 /*batch_Size*/, cfg.context_length, cfg.context_length);

    auto optimizer = torch::optim::AdamW(gpt2Mhl->parameters(),torch::optim::AdamWOptions(0.0004).weight_decay(0.1));
    int num_epochs = 10;
    EntropyData data2 = train_model_simple(gpt2Mhl, data,
        trainLoader,
        valLoader,
        optimizer,
        device, num_epochs, 5, 5, "every effort moves you",data.getTokenizer());

}
