//
// Created by moinshaikh on 7/10/26.
//

#include <Gpt2Untrained/dataPrepration.hpp>
#include <GPT2LargeLanguageModel/Gpt2Model.hpp>
#include <doctest.hpp>

// Equivalent of Python's generate_text_simple
torch::Tensor generateTextSimple(
    Gpt2& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size
) {
    for (int step = 0; step < max_new_tokens; ++step) {
        // Crop context if it exceeds the supported context size
        // idx.size(1) is int64_t, so cast 0 to match
        auto start = std::max(static_cast<int64_t>(0), idx.size(1) - context_size);
        torch::Tensor idx_cond = idx.slice(/*dim=*/1, /*start=*/start);

        // Get the predictions (no_grad scope)
        torch::NoGradGuard no_grad;
        torch::Tensor logits = model->forward(idx_cond);

        // Focus only on the last time step: [batch, n_tokens, vocab_size] -> [batch, vocab_size]
        logits = logits.select(/*dim=*/1, /*index=*/-1);

        // Apply softmax to get probabilities
        torch::Tensor probas = torch::softmax(logits, /*dim=*/-1);

        // Greedy: get the token with the highest probability
        torch::Tensor idx_next = torch::argmax(probas, /*dim=*/-1, /*keepdim=*/true);

        // Append to running sequence
        idx = torch::cat({idx, idx_next}, /*dim=*/1);
    }

    return idx;
}

TEST_CASE("untrainedModelTests") {
    PreparedData data("/home/moinshaikh/CLionProjects/LargeLanguageModelcpp/datasets/gpt2.tiktoken");

    // Encode a starting prompt into a batch tensor [1, n_tokens]
    auto batch = data.encodeBatch({"Every effort moves you"});

    config cfg;
    Gpt2 model(cfg);

    // Show model output for the initial forward pass
    auto out = model->forward(batch);
    std::cout << "Initial output shape: " << out.sizes() << std::endl;

    // Generate 5 new tokens greedily from the starting prompt
    torch::Tensor generated = generateTextSimple(
        model,
        batch,          // starting tokens [1, 4]
        5,              // generate 5 new tokens
        cfg.context_length  // context window (1024)
    );

    std::cout << "Generated sequence shape: " << generated.sizes() << std::endl;
    std::cout << "Generated tokens: " << generated << std::endl;

    // Decode back to text
    std::string decodedText = data.decode(generated);
    std::cout << "Decoded text: " << decodedText << std::endl;


}
