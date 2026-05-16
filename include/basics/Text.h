//
// Created by moinshaikh on 5/10/26.
//

#ifndef LARGELANGUAGEMODELCPP_WORKINGWITHTEXT_H
#define LARGELANGUAGEMODELCPP_WORKINGWITHTEXT_H
#include<filesystem>
#include<iostream>
#include<vector>
#include<regex>
#include<fstream>
struct textMetaData
{
    std::string  textRaw;
    std::filesystem::path path;
    std::ifstream textFile;
};

class Text
{
private:
    textMetaData metaData;
public:
    Text(const std::filesystem::path &path);
    ~Text();

    void printText(unsigned int indexFrom,unsigned int indexTo);

    std::string getText()
    {
        return metaData.textRaw;
    }
};






#endif //LARGELANGUAGEMODELCPP_WORKINGWITHTEXT_H
