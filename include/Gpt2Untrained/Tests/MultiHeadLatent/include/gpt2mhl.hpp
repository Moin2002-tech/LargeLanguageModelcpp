//
// Created by moinshaikh on 8/10/26.
//

#ifndef LARGELANGUAGEMODELCPP_GPT2MHL_HPP
#define LARGELANGUAGEMODELCPP_GPT2MHL_HPP


#include<Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/TransformBlockV5.hpp>

class gpt2mhlImpl : public torch::nn::Module {
private:
    config cfg;
    torch::nn::Embedding tok_emb{nullptr}, pos_emb{nullptr};
    torch::nn::Dropout drop_emb;
    torch::nn::ModuleList trf_block{nullptr};
    int current_pos = 0;
    LayerNormV3 finalNorm{nullptr};
    torch::nn::Linear outHead{nullptr};
public:
    gpt2mhlImpl(config &cfg);
    torch::Tensor forward(torch::Tensor x, bool use_Cache = false);
    int64_t getContextLength() const { return pos_emb->options.num_embeddings(); }
    void reset_kv_cache()
    {

        for (int i = 0; i < static_cast<int>(trf_block->size()); ++i) {
            auto blk = trf_block->ptr<TransformBlockV5Impl>(i);
            blk->reset_cache();
        }
        // self.ptr_current_pos = 0
        current_pos = 0;
    }

    auto getpos_emb() {
        return pos_emb;
    }

};TORCH_MODULE(gpt2mhl);


#endif //LARGELANGUAGEMODELCPP_GPT2MHL_HPP
