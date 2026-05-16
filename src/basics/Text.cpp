//
// Created by moinshaikh on 5/11/26.
//

#include"basics/Text.h"
#include<fstream>

Text::Text(const std::filesystem::path &path)
{
    namespace fs = std::filesystem;
    metaData.path = path;
    if (!fs::exists(path))
    {
        std::cout<<"the path is wrong";
    }

    metaData.textFile.open(path);
}

Text::~Text() {
    metaData.textFile.close();
}

void Text::printText(unsigned int indexFrom,unsigned int indexTo)
{
    if (metaData.textFile.is_open())
        {
        std::string content((std::istreambuf_iterator<char>(metaData.textFile)),
                           std::istreambuf_iterator<char>());
        metaData.textRaw = content;
        
        if (indexFrom < content.length() && indexTo <= content.length() && indexFrom < indexTo)
        {
            std::cout << content.substr(indexFrom, indexTo - indexFrom) << std::endl;
        }
        else
        {
            std::cout << content << std::endl;
        }
    }
}
