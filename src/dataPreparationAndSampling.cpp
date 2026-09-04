#include "ATen/ops/max.h"
#include "basics/GPTDatasetV1.h"
#include "tiktoken.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <torch/torch.h>
#include <vector>
#include "../external/third_party/doctest.hpp"

// Helper function to create a dataloader
// In C++, we return a unique_ptr to the dataloader.
// We use a template for the Sampler if we want to be generic, but
auto create_dataloader(const std::string &txt,
                       std::shared_ptr<tiktoken::Encoding> tokenizer,
                       int batch_size = 4, int max_length = 256,
                       int stride = 128) {

  // Create dataset and apply the Stack transform to collate tensors into
  // batches
  auto dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
                     .map(torch::data::transforms::Stack<>());

  // Create dataloader
  // SequentialSampler is used to match the default behavior of Python's
  // DataLoader (shuffle=False)
  auto dataloader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(dataset),
          torch::data::DataLoaderOptions().batch_size(batch_size));

  return dataloader;
}

TEST_CASE("DataPreparation&Sampling") {
  try
  {
    // 1. Initialize the tokenizer (cl100k_base as in the workspace)
    auto ranks = tiktoken::load_tiktoken_bpe_from_file(
        std::string(DATASETS_DIR) + "cl100k_base.tiktoken");
    tiktoken::EncodingDefinition def;
    def.name = "cl100k_base";
    def.pat_str =
        R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s)";
    def.mergeable_ranks = std::move(ranks);
    def.special_tokens = {{"<|endoftext|>", 100257}};
    tiktoken::register_encoding(def);
    auto tokenizer = tiktoken::get_encoding("cl100k_base");

    // 2. Read the text file
    std::ifstream file(std::string(DATASETS_DIR) + "the-verdict.txt");
    if (!file.is_open()) {
      std::cerr << "Could not open " << std::string(DATASETS_DIR) << "the-verdict.txt" << std::endl;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string raw_text = buffer.str();

    // 3. Create dataloader (using smaller sizes for demonstration)
    int batch_size = 8;
    int max_length = 4;
    int stride = 4;
    auto dataloader =
        create_dataloader(raw_text, tokenizer, batch_size, max_length, stride);

    // 4. Iterate through the dataloader to verify
    std::cout << "Iterating through dataloader..." << std::endl;
    for (auto &batch : *dataloader) {
      auto input = batch.data;
      auto target = batch.target;
      // print first 4 input and target
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

      break; // Just one batch for verification
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  torch::Tensor index = torch::tensor({2, 3, 1});
  auto numIndex = torch::max(index).item<int>() + 1;
  auto embed = torch::nn::Embedding(numIndex, 5);
  std::cout << "Embedding: " << embed->weight << std::endl;





}
