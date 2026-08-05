//
// Created by moinshaikh on 7/9/26.
//

#ifndef LARGELANGUAGEMODELCPP_UTIL_HPP
#define LARGELANGUAGEMODELCPP_UTIL_HPP

#include <string>
#include <stdexcept>

struct config {
    int vocab_size = 50257;    // Vocabulary size
    int context_length = 1024; // Context length
     int emb_dim = 768 ;      // Embedding dimension
     int n_heads = 12  ;      //Number of attention heads
     int n_layer =  12 ;      // Number of layers
     float drop_rate =0.1;      // # Dropout rate
     bool qkv_bias = false;     //  # Query-Key-Value bias
    int kv_window_size = 1024;
    int n_kv_groups = 2;
};

//
// Returns a config for the given GPT-2 model name.
// Supported names: "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"
//
inline config get_gpt2_config(const std::string& model_name)
{
    config cfg;

    if (model_name == "gpt2-small")
    {
        cfg.emb_dim  = 768;
        cfg.n_layer  = 12;
        cfg.n_heads  = 12;
    }
    else if (model_name == "gpt2-medium")
    {
        cfg.emb_dim  = 1024;
        cfg.n_layer  = 24;
        cfg.n_heads  = 16;
    }
    else if (model_name == "gpt2-large")
    {
        cfg.emb_dim  = 1280;
        cfg.n_layer  = 36;
        cfg.n_heads  = 20;
    }
    else if (model_name == "gpt2-xl")
    {
        cfg.emb_dim  = 1600;
        cfg.n_layer  = 48;
        cfg.n_heads  = 25;
    }
    else
    {
        throw std::invalid_argument("Incorrect model name " + model_name);
    }

    return cfg;
}

#endif //LARGELANGUAGEMODELCPP_UTIL_HPP