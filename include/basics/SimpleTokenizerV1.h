//
// Created by moinshaikh on 5/11/26.
//

#ifndef LARGELANGUAGEMODELCPP_SIMPLETOKENIZERV1_H
#define LARGELANGUAGEMODELCPP_SIMPLETOKENIZERV1_H
#include<regex>
#include<iostream>
#include<unordered_map>
#include<string>
#include<vector>

class SimpleTokenizerV1
{
private:
    std::unordered_map<std::string, int> str_to_int;
    std::unordered_map<int, std::string> int_to_str;
    std::regex token_regex;

public:
    explicit SimpleTokenizerV1(const std::unordered_map<std::string, int>& vocab);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};

#endif //LARGELANGUAGEMODELCPP_SIMPLETOKENIZERV1_H
