//
// Created by moinshaikh on 5/11/26.
//

#ifndef LARGELANGUAGEMODELCPP_TOKENIZE_H
#define LARGELANGUAGEMODELCPP_TOKENIZE_H

#include "Text.h"
#include<vector>
#include<regex>
#include<string>


class Tokenize
{
private:
    std::vector<std::string> result;
    std::string text;
public:
    Tokenize(){}
    ~Tokenize(){}
    std::vector<std::string> tokenize(Text &textObj, const std::regex &regex);
    std::string getText()
    {
        return text;
    }

    std::vector<std::string> getTokens()
    {
        return result;
    }




};

#endif //LARGELANGUAGEMODELCPP_TOKENIZE_H
