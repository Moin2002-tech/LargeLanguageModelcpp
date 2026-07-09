//
// Created by moinshaikh on 7/9/26.
//

#ifndef LARGELANGUAGEMODELCPP_DATAPREPRATION_HPP
#define LARGELANGUAGEMODELCPP_DATAPREPRATION_HPP

#include<string>
#include<tiktoken.hpp>
#include<torch/torch.h>
#include<vector>
#include<memory>

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
};

#endif //LARGELANGUAGEMODELCPP_DATAPREPRATION_HPP