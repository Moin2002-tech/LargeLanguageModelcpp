//
// Created by moinshaikh on 7/10/26.
//

#include <Gpt2Untrained/dataPreparation.hpp>
#include <GPT2LargeLanguageModel/Gpt2Model.hpp>
#include <doctest.hpp>





TEST_CASE("untrainedModelTests") {
    PreparedData data(MODELS_DIR "/gpt2.tiktoken");

    // Encode a starting prompt into a batch tensor [1, n_tokens]
    auto batch = data.encodeBatch({"Every effort moves you"});

    config cfg;
    Gpt2 model(cfg);

    // Show model output for the initial forward pass
    auto out = model->forward(batch);
    std::cout << "Initial output shape: " << out.sizes() << std::endl;

    // Generate 5 new tokens greedily from the starting prompt
    torch::Tensor generated = data.generateTextSimple(
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