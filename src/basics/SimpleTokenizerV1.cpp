//
// Created by moinshaikh on 5/13/26.
//

#include "basics/SimpleTokenizerV1.h"

SimpleTokenizerV1::SimpleTokenizerV1(const std::unordered_map<std::string, int>& vocab)
    : str_to_int(vocab), token_regex(R"(([,.:;?_!"()\']|--|\s))")
{
    // Build the reverse mapping from int to string
    for (const auto& pair : vocab) {
        int_to_str[pair.second] = pair.first;
    }
}

std::vector<int> SimpleTokenizerV1::encode(const std::string& text) {
    std::vector<int> ids;
    
    // Split text using regex
    std::sregex_token_iterator it(text.begin(), text.end(), token_regex, {-1, 1});
    std::sregex_token_iterator end;
    
    std::vector<std::string> preprocessed;
    for (; it != end; ++it) {
        std::string token = it->str();
        // Only add non-empty tokens after stripping
        if (!token.empty()) {
            // Strip whitespace from the token
            size_t start = token.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                size_t end_pos = token.find_last_not_of(" \t\n\r");
                token = token.substr(start, end_pos - start + 1);
                if (!token.empty()) {
                    preprocessed.push_back(token);
                }
            }
        }
    }
    
    // Convert tokens to IDs
    for (const auto& token : preprocessed) {
        auto it = str_to_int.find(token);
        if (it != str_to_int.end()) {
            ids.push_back(it->second);
        } else {
            // Handle unknown tokens - you might want to throw an exception or use a special token
            std::cerr << "Warning: Unknown token '" << token << "'" << std::endl;
        }
    }
    
    return ids;
}

std::string SimpleTokenizerV1::decode(const std::vector<int>& ids) {
    std::string text;
    
    // Convert IDs back to strings with spaces
    for (size_t i = 0; i < ids.size(); ++i) {
        auto it = int_to_str.find(ids[i]);
        if (it != int_to_str.end()) {
            text += it->second;
            if (i < ids.size() - 1) {
                text += " ";
            }
        } else {
            std::cerr << "Warning: Unknown ID " << ids[i] << std::endl;
        }
    }
    
    // Remove spaces before punctuation using regex
    text = std::regex_replace(text, std::regex(R"(\s+([,.?!"()\']))"), "$1");
    
    return text;
}
