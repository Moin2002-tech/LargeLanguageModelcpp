#include<TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

TextManager::TextManager(std::size_t maxSizeMb,
                          std::string separator,
                          double englishThreshold,
                          std::string fallbackEncoding)
    : maxSizeBytes_(maxSizeMb * 1024ULL * 1024ULL)
    , separator_(std::move(separator))
    , englishThreshold_(englishThreshold)
    , fallbackEncoding_(std::move(fallbackEncoding))
{
}

void TextManager::countCodepoints(const std::string& utf8Text,
                                   std::size_t& totalCodepoints,
                                   std::size_t& asciiCodepoints)
{
    totalCodepoints = 0;
    asciiCodepoints = 0;

    std::size_t i = 0;
    const std::size_t n = utf8Text.size();
    while (i < n)
    {
        const unsigned char byte0 = static_cast<unsigned char>(utf8Text[i]);
        std::size_t sequenceLength = 1;
        unsigned int codepoint = byte0;

        if ((byte0 & 0x80U) == 0x00U)
        {
            sequenceLength = 1;
            codepoint = byte0;
        }
        else if ((byte0 & 0xE0U) == 0xC0U && i + 1 < n)
        {
            sequenceLength = 2;
            codepoint = byte0 & 0x1FU;
        }
        else if ((byte0 & 0xF0U) == 0xE0U && i + 2 < n)
        {
            sequenceLength = 3;
            codepoint = byte0 & 0x0FU;
        }
        else if ((byte0 & 0xF8U) == 0xF0U && i + 3 < n)
        {
            sequenceLength = 4;
            codepoint = byte0 & 0x07U;
        }
        else
        {
            // Malformed byte: treat as a single replacement codepoint so
            // counting still makes progress instead of looping forever.
            ++totalCodepoints;
            ++i;
            continue;
        }

        for (std::size_t k = 1; k < sequenceLength; ++k)
        {
            const unsigned char cont = static_cast<unsigned char>(utf8Text[i + k]);
            codepoint = (codepoint << 6) | (cont & 0x3FU);
        }

        ++totalCodepoints;
        if (codepoint < 128U)
        {
            ++asciiCodepoints;
        }
        i += sequenceLength;
    }
}

bool TextManager::isValidUtf8(const std::string& bytes)
{
    std::size_t i = 0;
    const std::size_t n = bytes.size();
    while (i < n)
    {
        const unsigned char b0 = static_cast<unsigned char>(bytes[i]);
        std::size_t len = 0;

        if ((b0 & 0x80U) == 0x00U) len = 1;
        else if ((b0 & 0xE0U) == 0xC0U) len = 2;
        else if ((b0 & 0xF0U) == 0xE0U) len = 3;
        else if ((b0 & 0xF8U) == 0xF0U) len = 4;
        else return false;

        if (i + len > n) return false;

        for (std::size_t k = 1; k < len; ++k)
        {
            const unsigned char bk = static_cast<unsigned char>(bytes[i + k]);
            if ((bk & 0xC0U) != 0x80U) return false;
        }
        i += len;
    }
    return true;
}

std::string TextManager::decodeLatin1ToUtf8(const std::string& bytes)
{
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char b : bytes)
    {
        if (b < 0x80U)
        {
            out.push_back(static_cast<char>(b));
        }
        else
        {
            // Encode Latin-1 codepoint (0x80-0xFF) as 2-byte UTF-8.
            out.push_back(static_cast<char>(0xC0U | (b >> 6)));
            out.push_back(static_cast<char>(0x80U | (b & 0x3FU)));
        }
    }
    return out;
}

bool TextManager::isEnglish(const std::string& text) const
{
    if (text.empty())
    {

        return false;
    }
    std::size_t total = 0;
    std::size_t ascii = 0;
    countCodepoints(text, total, ascii);
    if (total == 0) return false;
    return static_cast<double>(ascii) / static_cast<double>(total) > englishThreshold_;
}

std::string TextManager::stripHeaders(const std::string& text)
{
    static const std::regex startMarker(R"(\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*)",
                                         std::regex::icase);
    static const std::regex endMarker(R"(\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*)",
                                       std::regex::icase);

    std::smatch startMatch;
    std::string::const_iterator bodyBegin = text.cbegin();
    if (std::regex_search(text, startMatch, startMarker))
    {
        bodyBegin = startMatch[0].second;
        if (bodyBegin != text.cend() && *bodyBegin == '\n')
        {
            ++bodyBegin;
        }
    }

    std::string::const_iterator bodyEnd = text.cend();
    std::smatch endMatch;
    if (std::regex_search(bodyBegin, text.cend(), endMatch, endMarker))
    {
        bodyEnd = endMatch[0].first;
    }

    std::string body(bodyBegin, bodyEnd);

    // Trim leading/trailing whitespace-only lines left behind by the cut.
    const auto first = body.find_first_not_of(" \t\r\n");
    const auto last = body.find_last_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }
    return body.substr(first, last - first + 1);
}

std::string TextManager::collapseBlankLines(const std::string& text)
{
    static const std::regex blankRun(R"(\n\s*\n)");
    return std::regex_replace(text, blankRun, "\n\n");
}

std::string TextManager::readFileWithFallback(const std::string& path) const
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string raw = buffer.str();

    if (isValidUtf8(raw))
    {
        return raw;
    }

    std::cerr << "Warning: UnicodeDecodeError encountered. "
                 "Trying fallback encoding for " << path << "\n";
    return decodeLatin1ToUtf8(raw);
}

std::vector<std::string> TextManager::collectFiles(const std::string& dataDir,
                                                     const std::vector<std::string>& extensions)
{
    std::vector<std::string> result;
    if (!fs::exists(dataDir))
    {
        return result;
    }

    for (const auto& entry : fs::recursive_directory_iterator(dataDir))
    {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        for (const auto& ext : extensions)
        {
            if (name.size() >= ext.size() &&
                name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
            {
                result.push_back(entry.path().string());
                break;
            }
        }
    }
    return result;
}

int TextManager::combineFiles(const std::vector<std::string>& filePaths,
                               const std::string& targetDir) const
{
    if (!fs::exists(targetDir))
    {
        fs::create_directories(targetDir);
    }

    std::vector<std::string> currentContent;
    std::size_t currentSize = 0;
    int fileCounter = 1;

    auto flush = [&](int counter)
    {
        const fs::path targetFilePath = fs::path(targetDir) / ("combined_" + std::to_string(counter) + ".txt");
        std::ofstream out(targetFilePath, std::ios::binary);
        for (std::size_t i = 0; i < currentContent.size(); ++i)
        {
            if (i > 0) out << separator_;
            out << currentContent[i];
        }
    };

    std::size_t processed = 0;
    for (const auto& filePath : filePaths)
    {
        ++processed;
        std::string content = readFileWithFallback(filePath);

        if (!isEnglish(content))
        {
            std::cerr << "Skipping " << filePath
                       << " as it does not contain primarily English text.\n";
            continue;
        }

        content = stripHeaders(content);
        content = collapseBlankLines(content);

        const std::size_t estimatedSize = content.size(); // UTF-8 bytes

        if (currentSize + estimatedSize > maxSizeBytes_)
        {
            flush(fileCounter);
            ++fileCounter;
            currentContent.clear();
            currentContent.push_back(content);
            currentSize = estimatedSize;
        }
        else
        {
            currentContent.push_back(content);
            currentSize += estimatedSize;
        }
    }

    if (!currentContent.empty())
    {
        flush(fileCounter);
    }

    return fileCounter;
}