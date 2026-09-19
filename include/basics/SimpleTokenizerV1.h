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

// Custom transparent hasher for string-like keys
struct StringHash {
    using is_transparent = void; // Enables heterogeneous lookup

    std::size_t operator()(std::string_view sv) const noexcept
    {
        return std::hash<std::string_view>{}(sv);
    }
};

class SimpleTokenizerV1
{
private:
    std::unordered_map<std::string, int, StringHash, std::equal_to<>> str_to_int;
    std::unordered_map<int, std::string> int_to_str;
    std::regex token_regex;

public:
    explicit SimpleTokenizerV1(const std::unordered_map<std::string, int, StringHash, std::equal_to<>>& vocab);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
};

#endif //LARGELANGUAGEMODELCPP_SIMPLETOKENIZERV1_H
