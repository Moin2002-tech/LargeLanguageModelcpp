//
// Created by moinshaikh on 9/8/26.
//

#ifndef LARGELANGUAGEMODELCPP_GENERATETEXT_HPP
#define LARGELANGUAGEMODELCPP_GENERATETEXT_HPP
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/gpt2mhl.hpp>
inline torch::Tensor generate_with_temperature(gpt2mhl &model,
    torch::Tensor &idx,
    int max_new_tokens,
    int context_size,
    int top_k= 0,
    float temperature = 0.0,
    int eos_id = -1)
{
 for (int64_t i = 0; i < max_new_tokens; ++i)
     {
     std::cout << "iter " << i << " idx_cond len=" << idx.size(1) << std::endl;
        // idx_cond = idx[:, -context_size:]
        int64_t seq_len = idx.size(1);
        int64_t start_idx = std::max<int64_t>(0, seq_len - context_size);
        torch::Tensor idx_cond = idx.slice(/*dim=*/1, /*start=*/start_idx);

        torch::Tensor logits;
        {
            // with torch.no_grad():
            torch::NoGradGuard no_grad;
            logits = model(idx_cond);
        }

        // logits = logits[:, -1, :]
        logits = logits.select(/*dim=*/1, /*index=*/-1);

        // New: Filter logits with top_k sampling
        if (top_k > 0) {
            // Keep only top_k values
            auto topk_res = torch::topk(logits, top_k);
            torch::Tensor top_logits = std::get<0>(topk_res);

            // min_val = top_logits[:, -1]
            torch::Tensor min_val = top_logits.select(/*dim=*/-1, /*index=*/-1).unsqueeze(1);

            logits = torch::where(
                logits < min_val,
                torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                logits
            );
        }

        torch::Tensor idx_next;

        // New: Apply temperature scaling
        if (temperature > 0.0) {
            logits = logits / temperature;

            // numerical stability tip
            auto max_res = logits.max(/*dim=*/-1, /*keepdim=*/true);
            logits = logits - std::get<0>(max_res);

            // Apply softmax to get probabilities
            torch::Tensor probs = torch::softmax(logits, /*dim=*/-1);

            // Sample from the distribution
            idx_next = torch::multinomial(probs, /*num_samples=*/1);
        } else
        {
            // Otherwise same as before: get idx of the vocab entry with the highest logits value
            idx_next = torch::argmax(logits, /*dim=*/-1, /*keepdim=*/true);
        }

        // Stop generating early if end-of-sequence token is encountered
        if (eos_id >= 0) {
            if (idx_next.item<int64_t>() == eos_id) {
                break;
            }
        }

        // Same as before: append sampled index to the running sequence
        idx = torch::cat({idx, idx_next}, /*dim=*/1);
    }

    return idx;
}

#endif //LARGELANGUAGEMODELCPP_GENERATETEXT_HPP
