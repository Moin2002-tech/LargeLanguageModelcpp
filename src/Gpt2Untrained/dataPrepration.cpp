//
// Created by moinshaikh on 7/10/26.
//

#include <Gpt2Untrained/dataPrepration.hpp>

PreparedData::PreparedData(std::string_view modelPath) {
    // Load the BPE ranks from the .tiktoken file
    auto ranks = tiktoken::load_tiktoken_bpe_from_file(std::string(modelPath));

    // Define GPT-2 encoding
    tiktoken::EncodingDefinition def;
    def.name = "gpt2";
    def.pat_str = R"('(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s)";
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = {{"<|endoftext|>", 50256}};
    def.explicit_n_vocab = 50257;

    tiktoken::register_encoding(def);
    tokenizer = tiktoken::get_encoding("gpt2");
}

torch::Tensor PreparedData::encodeBatch(const std::vector<std::string>& texts, int pad_to_length) {
    if (texts.empty()) {
        throw std::invalid_argument("encodeBatch: texts vector is empty");
    }

    // First pass: encode all texts and find the max sequence length
    std::vector<std::vector<tiktoken::Rank>> allTokenIds;
    allTokenIds.reserve(texts.size());

    size_t maxLen = 0;
    for (const auto& text : texts) {
        auto tokenIds = tokenizer->encode(text);
        maxLen = std::max(maxLen, tokenIds.size());
        allTokenIds.push_back(std::move(tokenIds));
    }

    // Determine target sequence length
    size_t seqLen = (pad_to_length > 0) ? static_cast<size_t>(pad_to_length) : maxLen;

    // Second pass: pad/truncate and convert to tensors
    std::vector<torch::Tensor> tensors;
    tensors.reserve(texts.size());

    for (auto& tokenIds : allTokenIds) {
        if (tokenIds.size() >= seqLen) {
            // Truncate to seqLen
            std::vector<int64_t> ids(tokenIds.begin(), tokenIds.begin() + seqLen);
            tensors.push_back(torch::tensor(ids, torch::kInt64));
        } else {
            // Pad with 0 to seqLen
            std::vector<int64_t> ids(seqLen, 0);  // Fill all with 0 (pad token)
            for (size_t j = 0; j < tokenIds.size(); ++j) {
                ids[j] = static_cast<int64_t>(tokenIds[j]);
            }
            tensors.push_back(torch::tensor(ids, torch::kInt64));
        }
    }

    // Stack all tensors along dim=0 to create [batch_size, seq_len] tensor
    batch = torch::stack(tensors, 0);
    return batch;
}

std::string PreparedData::decode(torch::Tensor tokenIds) const {
    // Squeeze batch dim if present: [1, n] -> [n]
    if (tokenIds.dim() == 2 && tokenIds.size(0) == 1) {
        tokenIds = tokenIds.squeeze(0);
    }

    // Convert tensor to vector<Rank>
    auto accessor = tokenIds.accessor<int64_t, 1>();
    std::vector<tiktoken::Rank> ids;
    ids.reserve(tokenIds.size(0));
    for (int64_t i = 0; i < tokenIds.size(0); ++i) {
        ids.push_back(static_cast<tiktoken::Rank>(accessor[i]));
    }

    // Decode via tiktoken
    return tokenizer->decode(ids);
}
