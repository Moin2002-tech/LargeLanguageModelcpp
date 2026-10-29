//
// Created by moinshaikh on 8/10/26.
//
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/TransformBlockV5.hpp>
#include<Gpt2Untrained/Tests/MultiHeadLatent/include/MultiHeadLatentV1.hpp>

TransformBlockV5Impl::TransformBlockV5Impl(config &cfg) : cfg(cfg) {
    att = register_module("att",MultiHeadLatentV1(cfg.emb_dim, cfg.emb_dim, cfg.drop_rate, cfg.n_heads,  cfg.qkv_bias,cfg.latent_dim));
    ff = register_module("ff",FeedForwardV3(cfg));
    layerNorm1 = register_module("layerNorm1",LayerNormV3(cfg.emb_dim));
    layerNorm2 = register_module("layerNorm2",LayerNormV3(cfg.emb_dim));
    dropoutShortcut = register_module("dropout_shortcut", torch::nn::Dropout(torch::nn::DropoutOptions(cfg.drop_rate)));
}


torch::Tensor TransformBlockV5Impl::forward(torch::Tensor x, bool use_cache)
{
    torch::Tensor shortcut =x;
    x = layerNorm1(x);


    x = att(x,use_cache);

    x= dropoutShortcut(x);
    x = x + shortcut;


    shortcut = x;
    x = layerNorm2(x);
    x = ff(x);
    x = dropoutShortcut(x);
    x =  x+ shortcut;


    return x;

}

void TransformBlockV5Impl::reset_cache()
{
    att->reset_cache();
}

void TransformBlockV5Impl::load_pretrained(
const std::unordered_map<std::string,
torch::Tensor>& p, int b)
{
    const std::string h = "h" + std::to_string(b) + "__";

    att->load_pretrained(p.at(h + "attn__c_attn__w"), p.at(h + "attn__c_attn__b"),
                         p.at(h + "attn__c_proj__w"), p.at(h + "attn__c_proj__b"));

    ff->load_pretrained(p.at(h + "mlp__c_fc__w"),   p.at(h + "mlp__c_fc__b"),
                     p.at(h + "mlp__c_proj__w"), p.at(h + "mlp__c_proj__b"));
    layerNorm1->load_pretrained(p.at(h + "ln_1__g"), p.at(h + "ln_1__b"));
    layerNorm2->load_pretrained(p.at(h + "ln_2__g"), p.at(h + "ln_2__b"));
}