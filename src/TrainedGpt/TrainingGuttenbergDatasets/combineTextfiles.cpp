#include<TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.hpp>

#include <filesystem>
#include <iostream>
#include <string>

#include "doctest.hpp"
namespace fs = std::filesystem;
TEST_CASE("combine_files")
{
    auto envOr = [](const char* name, const std::string& fallback) -> std::string
    {
        const char* value = std::getenv(name);
        return value ? std::string(value) : fallback;
    };

    const std::string dataDir =
        envOr("TEXTMANAGER_DATA_DIR", std::string(DATASETS_DIR) + "guttenberg/gutenberg/data/raw");
    const std::string outputDir =
        envOr("TEXTMANAGER_OUTPUT_DIR", std::string(DATASETS_DIR) + "guttenberg_preprocessed");
    const std::size_t maxSizeMb =
        static_cast<std::size_t>(std::stoul(envOr("TEXTMANAGER_MAX_SIZE_MB", "500")));

    TextManager manager(maxSizeMb);

    const auto allFiles = TextManager::collectFiles(dataDir, {".txt", ".txt.utf8"});
    std::cout << allFiles.size() << " file(s) to process.\n";

    const int fileCounter = manager.combineFiles(allFiles, outputDir);

    std::cout << fileCounter << " file(s) saved in "
              << fs::absolute(outputDir).string() << "\n";

    // Sanity checks rather than hard assertions: the pipeline should never
    // report more combined files than input files plus one (the "extra
    // empty flush" edge case covered above is the worst case).
    if (!allFiles.empty())
    {
        CHECK(fileCounter >= 1);
        CHECK(static_cast<std::size_t>(fileCounter) <= allFiles.size() + 1);
    }
}
