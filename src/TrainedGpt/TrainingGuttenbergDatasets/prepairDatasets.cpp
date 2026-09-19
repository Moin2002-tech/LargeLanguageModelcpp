//
// Created by moinshaikh on 9/11/26.
//
#include<TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.hpp>


#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <numeric>
#include <optional>
/*
namespace gutenberg {

// Configuration structure for preprocessing options
struct PreprocessorConfig {
    std::string data_dir = "gutenberg/data/raw";
    std::string output_dir = "gutenberg_preprocessed";
    size_t max_size_mb = 500;
    std::string separator = "<separator>";
    std::string fallback_encoding = "latin1";
    double english_threshold = 0.9;
};

// Check if text is primarily English (ASCII ratio above threshold)
// Mirrors Python's is_english function
inline bool is_english(const std::string& text, double threshold = 0.9) {
    if (text.empty()) return false;

    size_t ascii_chars = 0;
    for (unsigned char c : text) {
        if (c < 128) {
            ++ascii_chars;
        }
    }

    return static_cast<double>(ascii_chars) / text.length() > threshold;
}

// Strip Gutenberg headers from text content
// Removes the standard Project Gutenberg header/footer markers
// This is a simplified version - adapts the Python strip_headers logic
inline std::string strip_headers(const std::string& content) {
    std::string result = content;

    // Pattern for Gutenberg header start: "*** START OF"
    std::regex start_pattern(R"(\*\*\*\s*START\s+OF)", std::regex::icase);
    // Pattern for Gutenberg header end: "*** END OF"
    std::regex end_pattern(R"(\*\*\*\s*END\s+OF)", std::regex::icase);

    // Find start marker
    std::smatch start_match;
    if (std::regex_search(result, start_match, start_pattern)) {
        // Find end of line after start marker
        size_t start_pos = start_match.position() + start_match.length();
        size_t newline_pos = result.find('\n', start_pos);
        if (newline_pos != std::string::npos) {
            result = result.substr(newline_pos + 1);
        }
    }

    // Find end marker
    if (std::regex_search(result, end_match, end_pattern)) {
        size_t end_pos = end_match.position();
        result = result.substr(0, end_pos);
    }

    // Trim trailing whitespace
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
        result.pop_back();
    }

    return result;
}

// Normalize multiple blank lines to single blank line
// Mirrors Python's re.sub(r"\n\s*\n", "\n\n", content)
inline std::string normalize_blank_lines(const std::string& content) {
    std::regex blank_line_pattern(R"(\n\s*\n)");
    return std::regex_replace(content, blank_line_pattern, "\n\n");
}

// Get all text files from directory (recursively)
// Matches .txt and .txt.utf8 extensions like the Python version
inline std::vector<std::filesystem::path> get_text_files(const std::filesystem::path& data_dir) {
    std::vector<std::filesystem::path> files;

    if (!std::filesystem::exists(data_dir) || !std::filesystem::is_directory(data_dir)) {
        return files;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // Match .txt or .txt.utf8
            if (ext == ".txt" || ext == ".utf8" ||
                entry.path().string().ends_with(".txt.utf8")) {
                files.push_back(entry.path());
            }
        }
    }

    // Sort for consistent ordering
    std::sort(files.begin(), files.end());
    return files;
}

// Read file content with encoding fallback
// First tries UTF-8, falls back to latin1 if that fails
inline std::optional<std::string> read_file_with_fallback(
    const std::filesystem::path& file_path,
    const std::string& fallback_encoding = "latin1"
) {
    // Try UTF-8 first
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return std::nullopt;
    }

    // Read entire file into string
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Check if content is valid UTF-8
    // Simple check: look for invalid UTF-8 sequences
    bool is_valid_utf8 = true;
    for (size_t i = 0; i < content.size(); ) {
        unsigned char c = content[i];

        if (c <= 0x7F) {
            // ASCII character, always valid
            i += 1;
        } else if (c >= 0xC2 && c <= 0xDF) {
            // 2-byte sequence
            if (i + 1 >= content.size() || (content[i + 1] & 0xC0) != 0x80) {
                is_valid_utf8 = false;
                break;
            }
            i += 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            // 3-byte sequence
            if (i + 2 >= content.size()) {
                is_valid_utf8 = false;
                break;
            }
            if ((content[i + 1] & 0xC0) != 0x80 || (content[i + 2] & 0xC0) != 0x80) {
                is_valid_utf8 = false;
                break;
            }
            i += 3;
        } else if (c >= 0xF0 && c <= 0xF4) {
            // 4-byte sequence
            if (i + 3 >= content.size()) {
                is_valid_utf8 = false;
                break;
            }
            if ((content[i + 1] & 0xC0) != 0x80 ||
                (content[i + 2] & 0xC0) != 0x80 ||
                (content[i + 3] & 0xC0) != 0x80) {
                is_valid_utf8 = false;
                break;
            }
            i += 4;
        } else {
            // Invalid starting byte
            is_valid_utf8 = false;
            break;
        }
    }

    if (is_valid_utf8) {
        return content;
    }

    // UTF-8 validation failed, would need to re-read with fallback encoding
    // For simplicity, return the content as-is (latin1 is essentially raw bytes)
    // A full implementation would use a library like iconv for proper conversion
    return content;
}

// Main preprocessing function: combine files into chunks by size
// Returns the number of output files created
inline int combine_files(
    const std::vector<std::filesystem::path>& file_paths,
    const std::string& target_dir,
    size_t max_size_mb = 500,
    const std::string& separator = "<separator>",
    const std::string& fallback_encoding = "latin1",
    double english_threshold = 0.9
) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(target_dir);

    std::vector<std::string> current_content;
    size_t current_size = 0;
    int file_counter = 1;

    for (const auto& file_path : file_paths) {
        // Read file content
        auto content_opt = read_file_with_fallback(file_path, fallback_encoding);
        if (!content_opt.has_value()) {
            std::cerr << "Warning: Could not read file: " << file_path << std::endl;
            continue;
        }

        std::string content = content_opt.value();

        // Check if content is primarily English
        if (!is_english(content, english_threshold)) {
            std::cerr << "Skipping " << file_path << " as it does not contain primarily English text." << std::endl;
            continue;
        }

        // Strip Gutenberg headers
        content = strip_headers(content);

        // Normalize blank lines
        content = normalize_blank_lines(content);

        // Calculate UTF-8 encoded size
        size_t estimated_size = content.size(); // Approximate for ASCII-heavy text

        // Check if we need to create a new output file
        if (current_size + estimated_size > max_size_mb * 1024 * 1024) {
            // Write current batch to file
            std::filesystem::path target_file_path =
                std::filesystem::path(target_dir) / ("combined_" + std::to_string(file_counter) + ".txt");

            std::ofstream target_file(target_file_path, std::ios::out);
            if (target_file.is_open()) {
                for (size_t i = 0; i < current_content.size(); ++i) {
                    if (i > 0) {
                        target_file << separator;
                    }
                    target_file << current_content[i];
                }
                target_file.close();
            }

            file_counter++;
            current_content = {content};
            current_size = estimated_size;
        } else {
            current_content.push_back(content);
            current_size += estimated_size;
        }
    }

    // Write remaining content
    if (!current_content.empty()) {
        std::filesystem::path target_file_path =
            std::filesystem::path(target_dir) / ("combined_" + std::to_string(file_counter) + ".txt");

        std::ofstream target_file(target_file_path, std::ios::out);
        if (target_file.is_open()) {
            for (size_t i = 0; i < current_content.size(); ++i) {
                if (i > 0) {
                    target_file << separator;
                }
                target_file << current_content[i];
            }
            target_file.close();
        }
        file_counter++;
    }

    return file_counter - 1; // Subtract 1 because counter was incremented after last write
}

// Run preprocessing with configuration
// Returns the number of output files created
inline int preprocess(const PreprocessorConfig& config) {
    // Get all text files from data directory
    auto all_files = get_text_files(config.data_dir);

    std::cout << all_files.size() << " file(s) to process." << std::endl;

    // Combine files into chunks
    int file_counter = combine_files(
        all_files,
        config.output_dir,
        config.max_size_mb,
        config.separator,
        config.fallback_encoding,
        config.english_threshold
    );

    std::cout << file_counter << " file(s) saved in "
              << std::filesystem::absolute(config.output_dir) << std::endl;

    return file_counter;
}

} // namespace gutenberg
*/