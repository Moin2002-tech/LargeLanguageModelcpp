#pragma once

#include "tiktoken.hpp"
#include <torch/data.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

class GPTDatasetV1 : public torch::data::Dataset<GPTDatasetV1> {
private:
    std::vector<torch::Tensor> input_ids;
    std::vector<torch::Tensor> target_ids;

public:
    GPTDatasetV1(const std::string& txt, std::shared_ptr<tiktoken::Encoding> tokenizer, int max_length, int stride) {
        // Tokenize the entire text
        auto token_ids = tokenizer->encode(txt, {"<|endoftext|>"});

        // Use a sliding window to chunk the book into overlapping sequences of max_length
        for (size_t i = 0; i + max_length < token_ids.size(); i += stride) {
            std::vector<int64_t> input_chunk;
            std::vector<int64_t> target_chunk;
            
            for (int j = 0; j < max_length; ++j) {
                input_chunk.push_back(static_cast<int64_t>(token_ids[i + j]));
                target_chunk.push_back(static_cast<int64_t>(token_ids[i + j + 1]));
            }

            input_ids.push_back(torch::tensor(input_chunk, torch::kInt64));
            target_ids.push_back(torch::tensor(target_chunk, torch::kInt64));
        }
    }

    // Override the get method to return a single example
    torch::data::Example<> get(size_t index) override {
        return {input_ids[index], target_ids[index]};
    }

    // Override the size method to return the number of examples
    torch::optional<size_t> size() const override {
        return input_ids.size();
    }
};