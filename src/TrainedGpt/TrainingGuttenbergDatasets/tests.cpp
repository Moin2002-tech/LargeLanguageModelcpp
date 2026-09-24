//
// Created by moinshaikh on 9/24/26.
//

#ifndef LARGELANGUAGEMODELCPP_TESTS_HPP
#define LARGELANGUAGEMODELCPP_TESTS_HPP
#include<doctest.hpp>
#include<TrainedGpt/TrainingGuttenbergDatasets/prepairDatasets.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace
{
    // Writes `content` to `path`, creating parent directories as needed.
    void writeFile(const fs::path& path, const std::string& content)
    {
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        out << content;
    }
}



TEST_CASE("non_ASCII")
{
    TextManager manager;
    // Repeated non-ASCII characters (Cyrillic) well below the 0.9 threshold.
    const std::string text = "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 a";
    CHECK(manager.isEnglish(text) == false);
}
TEST_CASE("EnglishOrNot")
{
    TextManager manager;
    CHECK(manager.isEnglish("The quick brown fox jumps over the lazy dog.") == true);
}
TEST_CASE("isEnglish: empty string does not crash and is not English")
{
    TextManager manager;
    CHECK(manager.isEnglish("") == false);
}

TEST_CASE("isEnglish: threshold is configurable")
{
    TextManager strict(500, "<|endoftext|>", 0.99);
    // ~90% ASCII: below a 0.99 threshold.
    const std::string text = "ascii ascii ascii \xC3\xA9"; // trailing 'é'
    CHECK(strict.isEnglish(text) == false);
}

TEST_CASE("collapseBlankLines: collapses multiple blank lines into one")
{
    const std::string input = "para one\n\n\n\npara two\n\n\npara three";
    const std::string expected = "para one\n\npara two\n\npara three";
    CHECK(TextManager::collapseBlankLines(input) == expected);
}

TEST_CASE("collapseBlankLines: collapses blank lines containing whitespace")
{
    const std::string input = "a\n \t \nb";
    const std::string expected = "a\n\nb";
    CHECK(TextManager::collapseBlankLines(input) == expected);
}

TEST_CASE("collapseBlankLines: text with no blank lines is unchanged")
{
    const std::string input = "line one\nline two\nline three";
    CHECK(TextManager::collapseBlankLines(input) == input);
}

TEST_CASE("stripHeaders: extracts body between Gutenberg start/end markers")
{
    const std::string input =
        "Some legal boilerplate here.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n"
        "Call me Ishmael.\n"
        "This is the actual book body.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n"
        "Trailing license text.\n";

    const std::string result = TextManager::stripHeaders(input);
    CHECK(result.find("Call me Ishmael.") != std::string::npos);
    CHECK(result.find("actual book body") != std::string::npos);
    CHECK(result.find("legal boilerplate") == std::string::npos);
    CHECK(result.find("Trailing license") == std::string::npos);
}

TEST_CASE("stripHeaders: text without markers is returned unchanged (trimmed)")
{
    const std::string input = "Just plain content with no Gutenberg markers.";
    CHECK(TextManager::stripHeaders(input) == input);
}

TEST_CASE("stripHeaders: handles 'THIS PROJECT GUTENBERG EBOOK' wording variant")
{
    const std::string input =
        "*** START OF THIS PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
        "Body text.\n"
        "*** END OF THIS PROJECT GUTENBERG EBOOK EXAMPLE ***\n";
    const std::string result = TextManager::stripHeaders(input);
    CHECK(result == "Body text.");
}

TEST_CASE("collectFiles: finds files with matching extensions recursively")
{
    const fs::path tempDir = fs::temp_directory_path() / "textmanager_test_collect";
    fs::remove_all(tempDir);

    writeFile(tempDir / "a.txt", "hello");
    writeFile(tempDir / "nested" / "b.txt.utf8", "world");
    writeFile(tempDir / "c.md", "ignored");

    const auto files = TextManager::collectFiles(tempDir.string(), {".txt", ".txt.utf8"});
    CHECK(files.size() == 2);

    fs::remove_all(tempDir);
}

TEST_CASE("collectFiles: missing directory returns an empty list")
{
    const auto files = TextManager::collectFiles("/path/does/not/exist", {".txt"});
    CHECK(files.empty());
}

TEST_CASE("readFileWithFallback: reads a plain UTF-8 file")
{
    const fs::path tempFile = fs::temp_directory_path() / "textmanager_test_utf8.txt";
    writeFile(tempFile, "hello world");

    TextManager manager;
    CHECK(manager.readFileWithFallback(tempFile.string()) == "hello world");

    fs::remove(tempFile);
}

TEST_CASE("readFileWithFallback: falls back to latin1-style decoding on invalid UTF-8")
{
    const fs::path tempFile = fs::temp_directory_path() / "textmanager_test_latin1.txt";
    // 0xE9 alone is not valid UTF-8 (it looks like the start of a 3-byte
    // sequence but has no continuation bytes).
    const std::string rawBytes = std::string("caf") + static_cast<char>(0xE9);
    writeFile(tempFile, rawBytes);

    TextManager manager;
    const std::string result = manager.readFileWithFallback(tempFile.string());
    // 0xE9 (latin1 'é') should decode to the 2-byte UTF-8 sequence C3 A9.
    CHECK(result == std::string("caf") + static_cast<char>(0xC3) + static_cast<char>(0xA9));

    fs::remove(tempFile);
}

TEST_CASE("combineFiles: concatenates English files with separator, skips non-English")
{
    const fs::path tempDir = fs::temp_directory_path() / "textmanager_test_combine";
    const fs::path srcDir = tempDir / "src";
    const fs::path outDir = tempDir / "out";
    fs::remove_all(tempDir);

    writeFile(srcDir / "one.txt", "This is a short English sentence for testing purposes.");
    writeFile(srcDir / "two.txt", "Another short English sentence used for testing here.");
    // Predominantly non-ASCII -> should be skipped.
    writeFile(srcDir / "three.txt", "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");

    TextManager manager(500, "<SEP>");
    const auto files = TextManager::collectFiles(srcDir.string(), {".txt"});
    const int counter = manager.combineFiles(files, outDir.string());

    CHECK(counter == 1);
    REQUIRE(fs::exists(outDir / "combined_1.txt"));

    std::ifstream in(outDir / "combined_1.txt", std::ios::binary);
    std::string combined((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    CHECK(combined.find("<SEP>") != std::string::npos);
    CHECK(combined.find("short English sentence") != std::string::npos);
    CHECK(combined.find("\xD0\x9F") == std::string::npos);

    fs::remove_all(tempDir);
}

TEST_CASE("combineFiles: starts a new output file once the size cap is exceeded")
{
    const fs::path tempDir = fs::temp_directory_path() / "textmanager_test_combine_cap";
    const fs::path srcDir = tempDir / "src";
    const fs::path outDir = tempDir / "out";
    fs::remove_all(tempDir);

    const std::string bigEnglishText(2000, 'x'); // long run of ASCII 'x' characters
    writeFile(srcDir / "a.txt", bigEnglishText);
    writeFile(srcDir / "b.txt", bigEnglishText);

    // A 0 MB cap means every file immediately exceeds it, so each file
    // ends up flushed into its own output file. This faithfully mirrors
    // the original Python's flush-then-accumulate logic, including its
    // edge case: the very first comparison already exceeds an empty
    // buffer, so an extra (empty) combined_1.txt is written before the
    // real content starts landing in combined_2.txt / combined_3.txt.
    TextManager manager(0 /* MB */, "<SEP>"); // 0 MB -> maxSizeBytes_ == 0
    const auto files = TextManager::collectFiles(srcDir.string(), {".txt"});
    const int counter = manager.combineFiles(files, outDir.string());

    CHECK(counter == 3);
    CHECK(fs::exists(outDir / "combined_1.txt"));
    CHECK(fs::exists(outDir / "combined_2.txt"));
    CHECK(fs::exists(outDir / "combined_3.txt"));
    CHECK(fs::file_size(outDir / "combined_1.txt") == 0);

    fs::remove_all(tempDir);
}
#endif //LARGELANGUAGEMODELCPP_TESTS_HPP
