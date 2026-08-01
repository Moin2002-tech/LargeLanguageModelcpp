// tiktoken.hpp - Public C++ API mirroring Python tiktoken around CoreBPE
#pragma once

#include <algorithm>
#include <cstdint>
#include <future>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "_tiktoken.hpp"

// This header provides a lightweight, header-only C++ API similar to Python's tiktoken.
// It wraps tiktoken::CoreBPE with a friendly Encoding class, simple registry, and
// minimal model-to-encoding helpers.

namespace tiktoken {

using Rank = tiktoken::Rank;
using U8 = tiktoken::U8;
using U8Vec = tiktoken::U8Vec;

// Forward declaration for local helper
inline U8Vec base64_decode(const std::string& s);

// A portable container of mergeable ranks and special tokens to construct an Encoding.
struct EncodingDefinition {
	std::string name;
	std::string pat_str;
	std::vector<std::pair<U8Vec, Rank>> mergeable_ranks;  // token bytes -> rank
	std::vector<std::pair<std::string, Rank>> special_tokens; // token string -> id
	std::optional<std::uint32_t> explicit_n_vocab{}; // for sanity checks
};

// Helpers to build the inputs expected by CoreBPE
inline EncoderMap make_encoder_map(const std::vector<std::pair<U8Vec, Rank>>& pairs) {
	EncoderMap m;
	m.reserve(pairs.size());
	for (const auto& kv : pairs) m.emplace(kv.first, kv.second);
	return m;
}

inline SpecialEncMap make_special_map(const std::vector<std::pair<std::string, Rank>>& pairs) {
	SpecialEncMap m;
	m.reserve(pairs.size());
	for (const auto& kv : pairs) m.emplace(kv.first, kv.second);
	return m;
}

// Encoding class: mirrors Python's tiktoken.core.Encoding for most common methods
class Encoding {
public:
	// Construct from definition
	explicit Encoding(const EncodingDefinition& def)
		: name_(def.name), pat_str_(def.pat_str),
		  mergeable_ranks_(def.mergeable_ranks),
		  special_tokens_(def.special_tokens),
		  max_token_value_(0) {
		// Sanity checks similar to Python
		for (const auto& kv : mergeable_ranks_) max_token_value_ = std::max(max_token_value_, kv.second);
		for (const auto& kv : special_tokens_) max_token_value_ = std::max(max_token_value_, kv.second);
		if (def.explicit_n_vocab) {
			const std::uint32_t n_vocab = *def.explicit_n_vocab;
			if (mergeable_ranks_.size() + special_tokens_.size() != n_vocab) {
				throw std::invalid_argument("EncodingDefinition explicit_n_vocab mismatch");
			}
			if (max_token_value_ != n_vocab - 1) {
				throw std::invalid_argument("EncodingDefinition max token id mismatch with explicit_n_vocab");
			}
		}

		core_ = std::make_shared<CoreBPE>(
			make_encoder_map(mergeable_ranks_),
			make_special_map(special_tokens_),
			pat_str_);

		// Cache special token set and value set
		for (const auto& kv : special_tokens_) {
			special_token_set_.insert(kv.first);
			special_token_values_.insert(kv.second);
		}
	}

	// Create directly from parts
	Encoding(std::string name,
			 std::string pat_str,
			 std::vector<std::pair<U8Vec, Rank>> mergeable_ranks,
			 std::vector<std::pair<std::string, Rank>> special_tokens,
			 std::optional<std::uint32_t> explicit_n_vocab = std::nullopt)
		: Encoding(EncodingDefinition{std::move(name), std::move(pat_str), std::move(mergeable_ranks), std::move(special_tokens), explicit_n_vocab}) {}

	const std::string& name() const { return name_; }
	std::uint32_t max_token_value() const { return max_token_value_; }
	std::uint32_t n_vocab() const { return max_token_value_ + 1; }

	// ==================== Encoding ====================
	std::vector<Rank> encode_ordinary(const std::string& text) const {
		return core_->encode_ordinary(text);
	}

	// Parallel encoding for large texts - splits text into chunks and encodes in parallel
	// Use this for texts > 8KB for better performance on multi-core systems
	std::vector<Rank> encode_ordinary_parallel(const std::string& text, std::size_t chunk_size = 4096) const {
		return core_->encode_ordinary_parallel(text, chunk_size);
	}

	// Encodes a string; only the tokens in allowed_special are treated as special.
	std::vector<Rank> encode(const std::string& text,
							 const std::set<std::string>& allowed_special = {}) const {
		return core_->encode(text, allowed_special).first;
	}

	// Batch encoding helpers (simple thread pool via async)
	std::vector<std::vector<Rank>> encode_ordinary_batch(const std::vector<std::string>& texts,
														 int num_threads = 8) const {
		return parallel_map<std::vector<Rank>>(texts, num_threads, [&](const std::string& s){
			return encode_ordinary(s);
		});
	}

	std::vector<std::vector<Rank>> encode_batch(const std::vector<std::string>& texts,
												int num_threads = 8,
												const std::set<std::string>& allowed_special = {}) const {
		return parallel_map<std::vector<Rank>>(texts, num_threads, [&](const std::string& s){
			return encode(s, allowed_special);
		});
	}

	// Unstable variant
	std::pair<std::vector<Rank>, std::vector<std::vector<Rank>>>
	encode_with_unstable(const std::string& text,
						 const std::set<std::string>& allowed_special = {}) const {
		return core_->encode_with_unstable(text, allowed_special);
	}

	// Single-piece helpers (no regex splitting)
	std::vector<Rank> encode_single_piece(const std::string& text) const {
		const U8* b = reinterpret_cast<const U8*>(text.data());
		return core_->encode_single_piece(b, text.size());
	}
	std::vector<Rank> encode_single_piece(const U8* bytes, std::size_t len) const {
		return core_->encode_single_piece(bytes, len);
	}
	Rank encode_single_token(const std::string& text_or_bytes) const {
		const U8* b = reinterpret_cast<const U8*>(text_or_bytes.data());
		return core_->encode_single_token(b, text_or_bytes.size());
	}
	Rank encode_single_token(const U8* bytes, std::size_t len) const {
		return core_->encode_single_token(bytes, len);
	}

	// ==================== Decoding ====================
	std::vector<U8> decode_bytes(const std::vector<Rank>& tokens) const {
		return core_->decode_bytes(tokens);
	}

	// Lossy: constructs a std::string from bytes. Invalid UTF-8 bytes are preserved as-is.
	std::string decode(const std::vector<Rank>& tokens) const {
		auto bytes = decode_bytes(tokens);
		return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	}

	std::vector<U8> decode_single_token_bytes(Rank token) const {
		std::vector<Rank> one{token};
		return decode_bytes(one);
	}

	std::vector<std::vector<U8>> decode_tokens_bytes(const std::vector<Rank>& tokens) const {
		std::vector<std::vector<U8>> out;
		out.reserve(tokens.size());
		for (auto t : tokens) out.push_back(decode_single_token_bytes(t));
		return out;
	}

	// Decodes text and returns offsets (in code units) for the start of each token
	std::pair<std::string, std::vector<std::size_t>>
	decode_with_offsets(const std::vector<Rank>& tokens) const {
		auto parts = decode_tokens_bytes(tokens);
		std::string text;
		text.reserve(64);
		std::vector<std::size_t> offsets;
		offsets.reserve(tokens.size());
		std::size_t text_len = 0;
		for (const auto& tokenBytes : parts) {
			// Approximate Python behaviour re: UTF-8 continuation bytes
			std::size_t add = 0;
			if (!tokenBytes.empty()) {
				U8 b0 = tokenBytes[0];
				bool cont = (0x80 <= b0 && b0 < 0xC0);
				offsets.push_back(text_len > 0 && cont ? text_len - 1 : text_len);
			} else {
				offsets.push_back(text_len);
			}
			text.append(reinterpret_cast<const char*>(tokenBytes.data()), tokenBytes.size());
			// Count UTF-8 leading bytes to increment logical length like Python implementation
			for (U8 b : tokenBytes) if (!(0x80 <= b && b < 0xC0)) ++text_len;
		}
		return {std::move(text), std::move(offsets)};
	}

	// ==================== Misc ====================
	std::set<std::string> special_tokens_set() const { return special_token_set_; }
	bool is_special_token(Rank token) const { return special_token_values_.count(token) > 0; }

	// Return byte values for all tokens, sorted by token id
	std::vector<U8Vec> token_byte_values() const { return core_->token_byte_values_sorted(); }

	// Convenience: try to get the end-of-text token id if present
	std::optional<Rank> eot_token() const {
		for (const auto& kv : special_tokens_) if (kv.first == std::string("<|endoftext|>")) return kv.second;
		return std::nullopt;
	}

private:
	template <class R, class Fn>
	static std::vector<R> parallel_map(const std::vector<std::string>& xs, int num_threads, Fn fn) {
		if (xs.empty()) return {};
		if (num_threads <= 0) num_threads = 1;
		const int T = std::min<int>(num_threads, std::max<int>(1, std::thread::hardware_concurrency()));
		std::vector<R> out(xs.size());
		std::vector<std::future<void>> futs;
		futs.reserve(T);
		auto worker = [&](int tid) {
			std::size_t N = xs.size();
			for (std::size_t i = tid; i < N; i += T) out[i] = fn(xs[i]);
		};
		for (int t = 0; t < T; ++t) futs.emplace_back(std::async(std::launch::async, worker, t));
		for (auto& f : futs) f.get();
		return out;
	}

	std::string name_;
	std::string pat_str_;
	std::vector<std::pair<U8Vec, Rank>> mergeable_ranks_;
	std::vector<std::pair<std::string, Rank>> special_tokens_;
	std::shared_ptr<CoreBPE> core_;
	std::uint32_t max_token_value_{};
	std::set<std::string> special_token_set_{};
	std::set<Rank> special_token_values_{};
};

// =============== Simple Registry (plugins-style) ===============
class Registry {
public:
	static Registry& instance() {
		static Registry r; return r;
	}

	void register_encoding(const EncodingDefinition& def) {
		std::lock_guard<std::mutex> lk(mu_);
		auto enc = std::make_shared<Encoding>(def);
		encodings_[def.name] = std::move(enc);
	}

	std::shared_ptr<Encoding> get_encoding(const std::string& name) const {
		std::lock_guard<std::mutex> lk(mu_);
		auto it = encodings_.find(name);
		if (it == encodings_.end()) throw std::invalid_argument("Unknown encoding: " + name);
		return it->second;
	}

	std::vector<std::string> list_encoding_names() const {
		std::lock_guard<std::mutex> lk(mu_);
		std::vector<std::string> names;
		names.reserve(encodings_.size());
		for (auto& kv : encodings_) names.push_back(kv.first);
		return names;
	}

private:
	Registry() = default;
	mutable std::mutex mu_;
	std::unordered_map<std::string, std::shared_ptr<Encoding>> encodings_;
};

inline void register_encoding(const EncodingDefinition& def) { Registry::instance().register_encoding(def); }
inline std::shared_ptr<Encoding> get_encoding(const std::string& name) { return Registry::instance().get_encoding(name); }
inline std::vector<std::string> list_encoding_names() { return Registry::instance().list_encoding_names(); }

// =============== Model mapping helpers (subset) ===============
// Mirrors Python MODEL_PREFIX_TO_ENCODING and MODEL_TO_ENCODING
inline const std::unordered_map<std::string, std::string>& model_prefix_to_encoding() {
	static const std::unordered_map<std::string, std::string> M = {
		{"o1-", "o200k_base"},
		{"o3-", "o200k_base"},
		{"o4-mini-", "o200k_base"},
		{"gpt-5-", "o200k_base"},
		{"gpt-4.5-", "o200k_base"},
		{"gpt-4.1-", "o200k_base"},
		{"chatgpt-4o-", "o200k_base"},
		{"gpt-4o-", "o200k_base"},
		{"gpt-4-", "cl100k_base"},
		{"gpt-3.5-turbo-", "cl100k_base"},
		{"gpt-35-turbo-", "cl100k_base"},
		{"gpt-oss-", "o200k_harmony"},
		{"ft:gpt-4o", "o200k_base"},
		{"ft:gpt-4", "cl100k_base"},
		{"ft:gpt-3.5-turbo", "cl100k_base"},
		{"ft:davinci-002", "cl100k_base"},
		{"ft:babbage-002", "cl100k_base"},
	};
	return M;
}

inline const std::unordered_map<std::string, std::string>& model_to_encoding() {
	static const std::unordered_map<std::string, std::string> M = {
		{"o1", "o200k_base"},
		{"o3", "o200k_base"},
		{"o4-mini", "o200k_base"},
		{"gpt-4.1", "o200k_base"},
		{"gpt-4o", "o200k_base"},
		{"gpt-4", "cl100k_base"},
		{"gpt-3.5-turbo", "cl100k_base"},
		{"gpt-3.5", "cl100k_base"},
		{"gpt-35-turbo", "cl100k_base"},
		{"davinci-002", "cl100k_base"},
		{"babbage-002", "cl100k_base"},
		{"text-embedding-ada-002", "cl100k_base"},
		{"text-embedding-3-small", "cl100k_base"},
		{"text-embedding-3-large", "cl100k_base"},
		{"text-davinci-003", "p50k_base"},
		{"text-davinci-002", "p50k_base"},
		{"text-davinci-001", "r50k_base"},
		{"text-curie-001", "r50k_base"},
		{"text-babbage-001", "r50k_base"},
		{"text-ada-001", "r50k_base"},
		{"davinci", "r50k_base"},
		{"curie", "r50k_base"},
		{"babbage", "r50k_base"},
		{"ada", "r50k_base"},
		{"code-davinci-002", "p50k_base"},
		{"code-davinci-001", "p50k_base"},
		{"code-cushman-002", "p50k_base"},
		{"code-cushman-001", "p50k_base"},
		{"davinci-codex", "p50k_base"},
		{"cushman-codex", "p50k_base"},
		{"text-davinci-edit-001", "p50k_edit"},
		{"code-davinci-edit-001", "p50k_edit"},
		{"text-similarity-davinci-001", "r50k_base"},
		{"text-similarity-curie-001", "r50k_base"},
		{"text-similarity-babbage-001", "r50k_base"},
		{"text-similarity-ada-001", "r50k_base"},
		{"text-search-davinci-doc-001", "r50k_base"},
		{"text-search-curie-doc-001", "r50k_base"},
		{"text-search-babbage-doc-001", "r50k_base"},
		{"text-search-ada-doc-001", "r50k_base"},
		{"code-search-babbage-code-001", "r50k_base"},
		{"code-search-ada-code-001", "r50k_base"},
		{"gpt2", "gpt2"},
		{"gpt-2", "gpt2"},
	};
	return M;
}

inline std::string encoding_name_for_model(const std::string& model_name) {
	// Exact match
	auto it = model_to_encoding().find(model_name);
	if (it != model_to_encoding().end()) return it->second;
	// Prefix match
	for (const auto& kv : model_prefix_to_encoding()) {
		const auto& prefix = kv.first;
		if (model_name.rfind(prefix, 0) == 0) return kv.second;
	}
	throw std::invalid_argument("Could not map model to encoding: " + model_name +
								". Use register_encoding/get_encoding explicitly.");
}

inline std::shared_ptr<Encoding> encoding_for_model(const std::string& model_name) {
	return get_encoding(encoding_name_for_model(model_name));
}

// =============== Utilities to load .tiktoken bpe files ===============
// Parse a local .tiktoken file (base64 token bytes and integer rank per line)
// Returns vector of (tokenBytes, rank), suitable for EncodingDefinition.mergeable_ranks
inline std::vector<std::pair<U8Vec, Rank>> load_tiktoken_bpe_from_file(const std::string& path) {
	// Simple local file loader; no HTTP. One entry per line: b64 rank
	std::vector<std::pair<U8Vec, Rank>> out;
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("Failed to open file: " + path);
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		// split by space
		auto sp = line.find(' ');
		if (sp == std::string::npos) continue;
		std::string b64 = line.substr(0, sp);
		Rank r = static_cast<Rank>(std::stoul(line.substr(sp + 1)));
		// decode base64 (minimal implementation)
		U8Vec bytes = base64_decode(b64);
		out.emplace_back(std::move(bytes), r);
	}
	return out;
}

// Minimal base64 decode for the specific format used; no padding errors thrown.
inline U8Vec base64_decode(const std::string& s) {
	auto cvt = [](unsigned char c) -> int {
		if (c >= 'A' && c <= 'Z') return c - 'A';
		if (c >= 'a' && c <= 'z') return c - 'a' + 26;
		if (c >= '0' && c <= '9') return c - '0' + 52;
		if (c == '+') return 62;
		if (c == '/') return 63;
		if (c == '=') return -2; // padding
		return -1; // skip
	};
	U8Vec out; out.reserve(s.size()*3/4);
	int val = 0; int valb = -8;
	for (unsigned char c : s) {
		int d = cvt(c);
		if (d == -1) continue; // ignore whitespace/invalid
		if (d == -2) break;    // stop at padding
		val = (val << 6) | d; valb += 6;
		if (valb >= 0) { out.push_back(U8((val >> valb) & 0xFF)); valb -= 8; }
	}
	return out;
}

} // namespace tiktoken

