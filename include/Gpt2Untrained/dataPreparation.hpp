//
// Created by moinshaikh on 7/9/26.
//

#ifndef LARGELANGUAGEMODELCPP_DATAPREPARATION_HPP
#define LARGELANGUAGEMODELCPP_DATAPREPARATION_HPP

#include<string>
#include<tiktoken.hpp>
#include<torch/torch.h>
#include<vector>
#include<memory>
#include<GPT2LargeLanguageModel/Gpt2Model.hpp>
class PreparedData {
private:
    torch::Tensor batch;
    std::shared_ptr<tiktoken::Encoding> tokenizer;

public:
    explicit PreparedData(std::string_view modelPath);

    // Encode a vector of strings into a single stacked batch tensor [batch_size, seq_len]
    // pad_to_length: pad/truncate all sequences to this length (0 = no padding, use shortest sequence)
    torch::Tensor encodeBatch(const std::vector<std::string>& texts, int pad_to_length = 0);

    // Decode a flat tensor of token IDs back to a string
    // Equivalent to: tokenizer.decode(tokens.squeeze(0).tolist())
    std::string decode(torch::Tensor tokenIds) const;

    // Mirror Python text_to_token_ids:
    // encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    // return torch.tensor(encoded).unsqueeze(0)  -> [1, seq_len]
    torch::Tensor textToTokenIds(const std::string& text) const;

    //get Tokenizer
    auto getTokenizer() const  {
        return tokenizer;
    }
    auto getTokenizer()  {
        return tokenizer;
    }
    // Mirror Python token_ids_to_text:
    // flat = token_ids.squeeze(0); return tokenizer.decode(flat.tolist())
    std::string tokenIdsToText(torch::Tensor tokenIds) const;

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
};

#endif //LARGELANGUAGEMODELCPP_DATAPREPARATION_HPP