#pragma once

#include <cstddef>
#include <string>
#include <vector>

// TextManager
//
// C++ port of the Python "combine_files" preprocessing script used to
// prepare Project Gutenberg text files for LLM pretraining. It:
//   - walks a directory for raw text files,
//   - filters out files that are not primarily English/ASCII,
//   - strips Project Gutenberg boilerplate headers/footers,
//   - collapses runs of blank lines,
//   - concatenates files (joined by a separator token) into
//     size-capped "combined_N.txt" output files.


class TextManager
{
private:
    std::size_t maxSizeBytes_;
    std::string separator_;
    double englishThreshold_;
    std::string fallbackEncoding_; // documentation only; see readFileWithFallback

    // Counts total Unicode codepoints and how many of them are ASCII
    // (codepoint < 128). Mirrors Python iterating over decoded str characters.
    static void countCodepoints(const std::string& utf8Text,
                                 std::size_t& totalCodepoints,
                                 std::size_t& asciiCodepoints);

    // Returns true if `bytes` is well-formed UTF-8.
    static bool isValidUtf8(const std::string& bytes);

    // Reinterprets a raw byte string as if it had been decoded with a
    // single-byte "latin1"-style codec, producing valid UTF-8 output.
    // (Analogous to Python's `open(..., encoding="latin1")` fallback.)
    static std::string decodeLatin1ToUtf8(const std::string& bytes);

public:
    explicit TextManager(std::size_t maxSizeMb = 500,
                          std::string separator = "<|endoftext|>",
                          double englishThreshold = 0.9,
                          std::string fallbackEncoding = "latin1");

    // Returns true if the fraction of ASCII characters in `text` exceeds
    // the configured threshold.
    bool isEnglish(const std::string& text) const;

    // Strips Project Gutenberg's standard header/footer boilerplate,
    // returning only the body text between the START/END markers when
    // they are present.
    static std::string stripHeaders(const std::string& text);

    // Collapses any run of blank (whitespace-only) lines into a single
    // blank line, equivalent to Python's re.sub(r"\n\s*\n", "\n\n", text).
    static std::string collapseBlankLines(const std::string& text);

    // Reads a file as UTF-8; on decode failure, retries with the
    // configured fallback (latin1-style) decoding.
    std::string readFileWithFallback(const std::string& path) const;

    // Recursively collects paths under `dataDir` whose file name ends
    // with one of `extensions` (e.g. {".txt", ".txt.utf8"}).
    static std::vector<std::string> collectFiles(const std::string& dataDir,
                                                   const std::vector<std::string>& extensions);

    // Processes each file in `filePaths`: reads, filters non-English
    // content, strips headers, collapses blank lines, then accumulates
    // content into `targetDir/combined_<n>.txt` files, starting a new
    // file whenever the running size would exceed maxSizeMb. Returns the
    // number of combined files written (the highest file_counter used).
    int combineFiles(const std::vector<std::string>& filePaths,
                      const std::string& targetDir) const;

    // Accessors
    std::size_t maxSizeBytes() const { return maxSizeBytes_; }
    const std::string& separator() const { return separator_; }
    double englishThreshold() const { return englishThreshold_; }
};