#ifndef LARGELANGUAGEMODELCPP_GUTENBERG_PREPROCESSOR_HPP
#define LARGELANGUAGEMODELCPP_GUTENBERG_PREPROCESSOR_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>
#include <algorithm>
#include <numeric>

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
    bool is_english(const std::string& text, double threshold = 0.9);

    // Strip Gutenberg headers from text content
    // Removes the standard Project Gutenberg header/footer markers
    std::string strip_headers(const std::string& content);

    // Normalize multiple blank lines to single blank line
    std::string normalize_blank_lines(const std::string& content);

    // Get all text files from directory (recursively)
    std::vector<std::filesystem::path> get_text_files(const std::filesystem::path& data_dir);

    // Main preprocessing function: combine files into chunks by size
    // Returns the number of output files created
    int combine_files(
        const std::vector<std::filesystem::path>& file_paths,
        const std::string& target_dir,
        size_t max_size_mb = 500,
        const std::string& separator = "<separator>",
        const std::string& fallback_encoding = "latin1",
        double english_threshold = 0.9
    );

    // Run preprocessing with configuration
    // Returns the number of output files created
    int preprocess(const PreprocessorConfig& config);

} // namespace gutenberg

#endif // LARGELANGUAGEMODELCPP_GUTENBERG_PREPROCESSOR_HPP