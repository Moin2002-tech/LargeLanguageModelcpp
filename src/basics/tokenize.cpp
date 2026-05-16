//
// Created by moinshaikh on 5/11/26.
//


#include "basics/tokenize.h"





std::vector<std::string> Tokenize::tokenize(Text &textObj, const std::regex &regex)
{
    // Get text from Text object
    text = textObj.getText();
    
    // Clear previous results
    result.clear();

    // We use a regex_token_iterator
    // The {-1, 1} tells it to capture:
    // -1: The stuff between matches (the words)
    //  1: The first capturing group (the whitespace itself)
    std::sregex_token_iterator it(text.begin(), text.end(), regex, {-1, 1});
    std::sregex_token_iterator end;

    for (; it != end; ++it) {
        // Only add to result if the string isn't empty
        // (happens if there are leading/trailing delimiters)
        if (!it->str().empty()) {
            result.push_back(*it);
        }
    }

    // Print the result
    std::cout << "[";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << "'" << result[i] << "'" << (i == result.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    
    return result;
}
