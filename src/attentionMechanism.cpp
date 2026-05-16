//
// Created by moinshaikh on 5/15/26.
//
#include"../external/third_party/doctest.hpp"
#include<torch/torch.h>
#include<torch/nn.h>

class AttentionMechanismV1Impl : public torch::nn::Module
{
private:
    uint d_in, d_out;
    torch::Tensor W_query, W_key, W_value;

public:
    AttentionMechanismV1Impl(uint d_in, uint d_out) : d_in(d_in), d_out(d_out)
    {
        W_query = register_parameter("W_query", torch::rand({d_in, d_out}), false);
        W_key   = register_parameter("W_key",   torch::rand({d_in, d_out}), false);
        W_value = register_parameter("W_value", torch::rand({d_in, d_out}), false);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        // Compute Q, K, V from input
        auto Q = x.matmul(W_query);
        auto K = x.matmul(W_key);
        auto V = x.matmul(W_value);

        // Scaled dot-product attention
        auto attn_scores = Q.matmul(K.transpose(0, 1));
        auto attn_weights = torch::softmax(attn_scores / std::sqrt((double)d_out), 1);
        return attn_weights.matmul(V);
    }
};

TORCH_MODULE(AttentionMechanismV1);

template<typename T>
void print(const T &t)
{
    std::cout<<t<<"\n";
}
torch::Tensor softmax_naive(torch::Tensor &x) {
    return torch::exp(x) / torch::exp(x).sum(0);
}
TEST_CASE("AttentionMechanism")
{
    torch::Tensor input = torch::tensor({{0.43, 0.15, 0.89},//your
                                            {0.55, 0.87, 0.66}, //journey
                                            {0.57, 0.85, 0.64}, //start
                                            {0.22, 0.58, 0.33},//with
                                            {0.77, 0.25, 0.10},//one
                                            {0.05, 0.80, 0.55}//step
        });
    auto query = input[1];

    auto attn_scores_2 = torch::empty({input.size(0)});
    for (int i = 0; i < input.size(0); i++) {
        attn_scores_2[i] = torch::dot(input[i], query).item<float>();
    }
    std::cout << attn_scores_2 << std::endl;

    auto attn_scores_temp = attn_scores_2 / attn_scores_2.sum();
    std::cout << attn_scores_temp << std::endl;

    auto attnScoreNaive = softmax_naive(attn_scores_2);
    std::cout <<"attnScoreNaive= "<< attnScoreNaive << std::endl;
    std::cout <<"attnScoreNaiveSum= "<< attnScoreNaive.sum() << std::endl;

    auto attnScoreWeight = torch::softmax(attn_scores_2, 0);
    std::cout <<"attnScoreWeight= "<< attnScoreWeight << std::endl;
    std::cout <<"attnScoreWeightSum= "<< attnScoreWeight.sum() << std::endl;

    auto query2 = input[1];
    std::cout<<"query ="<< query2 << std::endl;
    auto context_vec2 = torch::zeros(query2.sizes());
    for (int i = 0; i < input.size(0); ++i) {
        context_vec2 += attnScoreWeight[i] * input[i];
    }
    std::cout<<"context_vec2 ="<< context_vec2 << std::endl;

    auto attentionScore2 = torch::empty({6,6});
    for (int i = 0; i < input.size(0); ++i) {
        for (int j = 0; j < input.size(0); ++j) {
            attentionScore2[i][j] = torch::dot(input[i], input[j]).item<float>();
        }
    }

    print("attentionScore2 = ");
    print(attentionScore2);

    auto x_2 = input[1];
    auto d_in= input.size(1);
    auto d_out = 2;

    torch::manual_seed(21);
    {
        AttentionMechanismV1 attention(d_in, d_out);
        auto output = attention->forward(input);
        std::cout << "AttentionMechanismV1 output = " << output << std::endl;
    }

    // Stand-alone: query_2 = x_2 @ W_query (like the Python snippet)

        torch::manual_seed(21);
        torch::Tensor W_query = torch::rand({d_in, d_out});
        torch::Tensor W_key   = torch::rand({d_in, d_out});
        torch::Tensor W_value = torch::rand({d_in, d_out});
        std::cout << "W tensors created: " << W_query.sizes() << ", " << W_key.sizes() << ", " << W_value.sizes() << std::endl;

        auto query_2 = x_2.matmul(W_query);
        auto key_2   = x_2.matmul(W_key);
        auto value_2 = x_2.matmul(W_value);

        std::cout << "query_2  = " << query_2  << std::endl;
        std::cout << "key_2    = " << key_2    << std::endl;
        std::cout << "value_2  = " << value_2  << std::endl;

    auto keys = input.matmul(W_key);
    auto values = input.matmul(W_value);
    auto queries = input.matmul(W_query);

}

