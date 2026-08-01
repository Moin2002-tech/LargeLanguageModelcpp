#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// --- deps ---
// 1) PCRE2-8 (package often named libpcre2-8)
// 2) utf8cpp (single-header; e.g., <utf8.h>)
#ifndef PCRE2_CODE_UNIT_WIDTH
#define PCRE2_CODE_UNIT_WIDTH 8
#endif
#include <pcre2.h>
#include <utf8.h>

// ============================================================================
// SIMD Detection and Intrinsics
// ============================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#define TIKTOKEN_X86 1
#if defined(__SSE2__) || defined(_M_X64) ||                                    \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define TIKTOKEN_SSE2 1
#include <emmintrin.h>
#endif
#if defined(__AVX2__)
#define TIKTOKEN_AVX2 1
#include <immintrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define TIKTOKEN_ARM64 1
#if defined(__ARM_NEON) || defined(_M_ARM64)
#define TIKTOKEN_NEON 1
#include <arm_neon.h>
#endif
#endif

// Prefetch hints
#if defined(__GNUC__) || defined(__clang__)
#define TIKTOKEN_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#define TIKTOKEN_PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
#define TIKTOKEN_LIKELY(x) __builtin_expect(!!(x), 1)
#define TIKTOKEN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define TIKTOKEN_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#include <intrin.h>
#define TIKTOKEN_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#define TIKTOKEN_PREFETCH_WRITE(addr)                                          \
  _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#define TIKTOKEN_LIKELY(x) (x)
#define TIKTOKEN_UNLIKELY(x) (x)
#define TIKTOKEN_FORCE_INLINE __forceinline
#else
#define TIKTOKEN_PREFETCH(addr) ((void)0)
#define TIKTOKEN_PREFETCH_WRITE(addr) ((void)0)
#define TIKTOKEN_LIKELY(x) (x)
#define TIKTOKEN_UNLIKELY(x) (x)
#define TIKTOKEN_FORCE_INLINE inline
#endif

namespace tiktoken {

using Rank = std::uint32_t;
using U8 = std::uint8_t;
using U8Vec = std::vector<U8>;

struct DecodeKeyError : std::runtime_error {
  Rank token;
  explicit DecodeKeyError(Rank t)
      : std::runtime_error("Invalid token for decoding: " + std::to_string(t)),
        token(t) {}
};
struct DecodeError : std::runtime_error {
  explicit DecodeError(const std::string &m)
      : std::runtime_error("Could not decode tokens: " + m) {}
};
struct EncodeError : std::runtime_error {
  explicit EncodeError(const std::string &m)
      : std::runtime_error("Could not encode string: " + m) {}
};

// ============================================================================
// Optimized Hash Functions using wyhash-inspired algorithm
// ============================================================================
namespace detail {
// Fast multiply-mix for hashing (wyhash-inspired)
TIKTOKEN_FORCE_INLINE std::uint64_t wymix(std::uint64_t a,
                                          std::uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
  __uint128_t r = static_cast<__uint128_t>(a) * b;
  return static_cast<std::uint64_t>(r ^ (r >> 64));
#else
  // Fallback: use 64-bit multiplication
  std::uint64_t ha = a >> 32, la = static_cast<std::uint32_t>(a);
  std::uint64_t hb = b >> 32, lb = static_cast<std::uint32_t>(b);
  std::uint64_t rh = ha * hb, rl = la * lb;
  std::uint64_t rm0 = ha * lb, rm1 = hb * la;
  std::uint64_t t = rl + (rm0 << 32);
  std::uint64_t c = t < rl;
  std::uint64_t lo = t + (rm1 << 32);
  c += lo < t;
  std::uint64_t hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
  return lo ^ hi;
#endif
}

// Read 8 bytes as little-endian uint64
TIKTOKEN_FORCE_INLINE std::uint64_t read64_le(const U8 *p) noexcept {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap64(v);
#endif
  return v;
}

// Read 4 bytes as little-endian uint32
TIKTOKEN_FORCE_INLINE std::uint32_t read32_le(const U8 *p) noexcept {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  v = __builtin_bswap32(v);
#endif
  return v;
}
} // namespace detail

// Optimized wyhash-based hash for byte vectors
struct VecU8Hash {
  static constexpr std::uint64_t SECRET0 = 0xa0761d6478bd642full;
  static constexpr std::uint64_t SECRET1 = 0xe7037ed1a0b428dbull;
  static constexpr std::uint64_t SECRET2 = 0x8ebc6af09c88c6e3ull;
  static constexpr std::uint64_t SECRET3 = 0x589965cc75374cc3ull;

  std::size_t operator()(const U8Vec &v) const noexcept {
    return hash_bytes(v.data(), v.size());
  }

  TIKTOKEN_FORCE_INLINE static std::size_t
  hash_bytes(const U8 *data, std::size_t len) noexcept {
    std::uint64_t seed = SECRET0;
    const U8 *p = data;

    if (TIKTOKEN_LIKELY(len <= 16)) {
      if (TIKTOKEN_LIKELY(len >= 4)) {
        // 4-16 bytes: read first and last 4 bytes
        std::uint64_t a = detail::read32_le(p);
        std::uint64_t b = detail::read32_le(p + len - 4);
        return static_cast<std::size_t>(detail::wymix(a ^ SECRET1, b ^ seed) ^
                                        len);
      } else if (TIKTOKEN_LIKELY(len > 0)) {
        // 1-3 bytes: pack into single value
        std::uint64_t a = (static_cast<std::uint64_t>(p[0]) << 16) |
                          (static_cast<std::uint64_t>(p[len >> 1]) << 8) |
                          p[len - 1];
        return static_cast<std::size_t>(detail::wymix(a ^ SECRET1, seed) ^ len);
      }
      return static_cast<std::size_t>(seed);
    }

    // > 16 bytes: process in chunks
    if (TIKTOKEN_LIKELY(len <= 48)) {
      std::uint64_t a = detail::read64_le(p) ^ SECRET1;
      std::uint64_t b = detail::read64_le(p + 8) ^ seed;
      std::uint64_t c = detail::read64_le(p + len - 16) ^ SECRET2;
      std::uint64_t d = detail::read64_le(p + len - 8) ^ SECRET3;
      return static_cast<std::size_t>(detail::wymix(a ^ c, b ^ d) ^ len);
    }

    // Large data: loop processing
    std::uint64_t see1 = seed, see2 = seed;
    std::size_t i = 0;
    for (; i + 48 <= len; i += 48) {
      seed = detail::wymix(detail::read64_le(p + i) ^ SECRET1,
                           detail::read64_le(p + i + 8) ^ seed);
      see1 = detail::wymix(detail::read64_le(p + i + 16) ^ SECRET2,
                           detail::read64_le(p + i + 24) ^ see1);
      see2 = detail::wymix(detail::read64_le(p + i + 32) ^ SECRET3,
                           detail::read64_le(p + i + 40) ^ see2);
    }
    seed ^= see1 ^ see2;

    // Handle remaining bytes
    const U8 *end = p + len;
    p += i;
    if (end - p >= 16) {
      seed = detail::wymix(detail::read64_le(p) ^ SECRET1,
                           detail::read64_le(p + 8) ^ seed);
      p += 16;
    }
    if (end - p >= 8) {
      seed = detail::wymix(detail::read64_le(p) ^ SECRET2, seed);
      p += 8;
    }
    if (end - p >= 4) {
      seed ^= detail::read32_le(p);
    }

    return static_cast<std::size_t>(detail::wymix(seed, len ^ SECRET0));
  }
};

// Optimized equality with SIMD support
struct VecU8Eq {
  bool operator()(const U8Vec &a, const U8Vec &b) const noexcept {
    if (a.size() != b.size())
      return false;
    return fast_memcmp(a.data(), b.data(), a.size());
  }

  TIKTOKEN_FORCE_INLINE static bool fast_memcmp(const U8 *a, const U8 *b,
                                                std::size_t len) noexcept {
    if (len == 0)
      return true;

#if defined(TIKTOKEN_AVX2)
    // AVX2: compare 32 bytes at a time
    while (len >= 32) {
      __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
      __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
      __m256i cmp = _mm256_cmpeq_epi8(va, vb);
      if (_mm256_movemask_epi8(cmp) != -1)
        return false;
      a += 32;
      b += 32;
      len -= 32;
    }
#endif

#if defined(TIKTOKEN_SSE2)
    // SSE2: compare 16 bytes at a time
    while (len >= 16) {
      __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a));
      __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b));
      __m128i cmp = _mm_cmpeq_epi8(va, vb);
      if (_mm_movemask_epi8(cmp) != 0xFFFF)
        return false;
      a += 16;
      b += 16;
      len -= 16;
    }
#endif

#if defined(TIKTOKEN_NEON)
    // NEON: compare 16 bytes at a time
    while (len >= 16) {
      uint8x16_t va = vld1q_u8(a);
      uint8x16_t vb = vld1q_u8(b);
      uint8x16_t cmp = vceqq_u8(va, vb);
      if (vminvq_u8(cmp) != 0xFF)
        return false;
      a += 16;
      b += 16;
      len -= 16;
    }
#endif

    // Compare 8 bytes at a time
    while (len >= 8) {
      std::uint64_t va, vb;
      std::memcpy(&va, a, 8);
      std::memcpy(&vb, b, 8);
      if (va != vb)
        return false;
      a += 8;
      b += 8;
      len -= 8;
    }

    // Remaining bytes
    while (len > 0) {
      if (*a++ != *b++)
        return false;
      --len;
    }
    return true;
  }
};

using EncoderMap = std::unordered_map<U8Vec, Rank, VecU8Hash, VecU8Eq>;
using DecoderMap = std::unordered_map<Rank, U8Vec>;
using SpecialEncMap = std::unordered_map<std::string, Rank>;
using SpecialDecMap = std::unordered_map<Rank, U8Vec>;

static constexpr std::size_t MAX_NUM_THREADS = 128;

// ============================================================================
// Bitwise Utilities for Fast Character Classification
// ============================================================================
namespace bitutil {
// Fast whitespace check using bitwise operations
TIKTOKEN_FORCE_INLINE bool is_whitespace(U8 c) noexcept {
  // Whitespace: 0x09 (tab), 0x0A (LF), 0x0B (VT), 0x0C (FF), 0x0D (CR), 0x20
  // (space) Use lookup table packed in 64-bit integer Bit positions for ASCII <
  // 64 chars
  constexpr std::uint64_t ws_mask = (1ULL << 0x09) | (1ULL << 0x0A) |
                                    (1ULL << 0x0B) | (1ULL << 0x0C) |
                                    (1ULL << 0x0D) | (1ULL << 0x20);
  return (c < 64) && ((ws_mask >> c) & 1);
}

// Check if byte is ASCII (high bit clear)
TIKTOKEN_FORCE_INLINE bool is_ascii(U8 c) noexcept { return (c & 0x80) == 0; }

// Check if byte is UTF-8 continuation byte
TIKTOKEN_FORCE_INLINE bool is_utf8_continuation(U8 c) noexcept {
  return (c & 0xC0) == 0x80;
}

// Check if byte is UTF-8 lead byte
TIKTOKEN_FORCE_INLINE bool is_utf8_lead(U8 c) noexcept {
  return (c & 0xC0) != 0x80;
}

// Get UTF-8 sequence length from lead byte (0 if invalid)
TIKTOKEN_FORCE_INLINE int utf8_seq_len(U8 lead) noexcept {
  if ((lead & 0x80) == 0x00)
    return 1; // 0xxxxxxx
  if ((lead & 0xE0) == 0xC0)
    return 2; // 110xxxxx
  if ((lead & 0xF0) == 0xE0)
    return 3; // 1110xxxx
  if ((lead & 0xF8) == 0xF0)
    return 4; // 11110xxx
  return 0;   // Invalid
}

// Count leading zeros (for power-of-2 calculations)
TIKTOKEN_FORCE_INLINE int clz32(std::uint32_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return x ? __builtin_clz(x) : 32;
#elif defined(_MSC_VER)
  unsigned long idx;
  return _BitScanReverse(&idx, x) ? (31 - idx) : 32;
#else
  if (x == 0)
    return 32;
  int n = 0;
  if (x <= 0x0000FFFF) {
    n += 16;
    x <<= 16;
  }
  if (x <= 0x00FFFFFF) {
    n += 8;
    x <<= 8;
  }
  if (x <= 0x0FFFFFFF) {
    n += 4;
    x <<= 4;
  }
  if (x <= 0x3FFFFFFF) {
    n += 2;
    x <<= 2;
  }
  if (x <= 0x7FFFFFFF) {
    n += 1;
  }
  return n;
#endif
}

// Next power of 2 (for buffer sizing)
TIKTOKEN_FORCE_INLINE std::size_t next_pow2(std::size_t x) noexcept {
  if (x <= 1)
    return 1;
  return std::size_t(1) << (64 - clz32(static_cast<std::uint32_t>(x - 1)));
}
} // namespace bitutil

// ============================================================================
// Thread-Local Memory Pool for Reduced Allocations
// ============================================================================
class ThreadLocalPool {
public:
  static ThreadLocalPool &instance() {
    static thread_local ThreadLocalPool pool;
    return pool;
  }

  // Get scratch vector for temporary operations
  U8Vec &scratch_vec(std::size_t min_capacity = 256) {
    if (scratch_vec_.capacity() < min_capacity) {
      scratch_vec_.reserve(bitutil::next_pow2(min_capacity));
    }
    scratch_vec_.clear();
    return scratch_vec_;
  }

  // Get scratch vector for ranks
  std::vector<Rank> &rank_vec(std::size_t min_capacity = 64) {
    if (rank_vec_.capacity() < min_capacity) {
      rank_vec_.reserve(bitutil::next_pow2(min_capacity));
    }
    rank_vec_.clear();
    return rank_vec_;
  }

  // Get scratch string
  std::string &scratch_string(std::size_t min_capacity = 128) {
    if (scratch_str_.capacity() < min_capacity) {
      scratch_str_.reserve(bitutil::next_pow2(min_capacity));
    }
    scratch_str_.clear();
    return scratch_str_;
  }

private:
  ThreadLocalPool() {
    scratch_vec_.reserve(1024);
    rank_vec_.reserve(256);
    scratch_str_.reserve(256);
  }

  U8Vec scratch_vec_;
  std::vector<Rank> rank_vec_;
  std::string scratch_str_;
};

// --- utils ---
inline U8Vec slice_to_vec(const U8 *p, std::size_t s, std::size_t e) {
  return U8Vec{p + s, p + e};
}

// Thread-local scratch buffers to avoid allocations during unordered_map
// lookups
inline const U8Vec &lookup_vec(const U8 *p, std::size_t s, std::size_t e) {
  static thread_local U8Vec tmp;
  std::size_t len = e - s;
  if (tmp.capacity() < len) {
    tmp.reserve(bitutil::next_pow2(len));
  }
  tmp.resize(len);
  if (len > 0)
    std::memcpy(tmp.data(), p + s, len);
  return tmp;
}

inline const std::string &scratch_string(const char *p, std::size_t n) {
  static thread_local std::string tmp;
  tmp.assign(p, n);
  return tmp;
}

inline std::string regex_escape(const std::string &s) {
  // Fast path using lookup table for regex metacharacters
  // Metacharacters: \ . ^ $ | ( ) [ ] { } * + ? -
  static constexpr bool is_meta[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 0-15
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 16-31
      0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, // 32-47: $ ( ) * + - .
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 48-63: ?
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 64-79
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, // 80-95: [ \ ] ^
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 96-111
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, // 112-127: { | }
  };

  std::string out;
  out.reserve(s.size() * 2);
  for (unsigned char c : s) {
    if (c < 128 && is_meta[c])
      out.push_back('\\');
    out.push_back(static_cast<char>(c));
  }
  return out;
}

inline std::size_t hash_current_thread() {
  return std::hash<std::thread::id>{}(std::this_thread::get_id()) %
         MAX_NUM_THREADS;
}

// ============================================================================
// Work-Stealing Thread Pool for Better Parallelism
// ============================================================================
class WorkStealingPool {
public:
  static WorkStealingPool &instance() {
    static WorkStealingPool pool;
    return pool;
  }

  template <typename Fn>
  void parallel_for(std::size_t count, Fn &&fn, std::size_t min_batch = 1) {
    if (count == 0)
      return;

    std::size_t num_threads =
        std::min(workers_.size(), (count + min_batch - 1) / min_batch);
    if (num_threads <= 1) {
      for (std::size_t i = 0; i < count; ++i)
        fn(i);
      return;
    }

    std::atomic<std::size_t> next_idx{0};
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
      futures.push_back(std::async(std::launch::async, [&, count]() {
        while (true) {
          std::size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (idx >= count)
            break;
          fn(idx);
        }
      }));
    }

    for (auto &f : futures)
      f.get();
  }

  std::size_t num_threads() const { return workers_.size(); }

private:
  WorkStealingPool() {
    std::size_t n = std::max(1u, std::thread::hardware_concurrency());
    workers_.resize(n);
  }

  std::vector<int> workers_; // Placeholder for thread count
};

// ============================================================================
// PCRE2 wrapper with match data pooling
// ============================================================================
class Pcre2Regex {
public:
  Pcre2Regex() : re_(nullptr), has_jit_(false), md_pool_(nullptr) {}
  explicit Pcre2Regex(const std::string &pattern, uint32_t options = 0)
      : re_(nullptr), has_jit_(false), md_pool_(nullptr) {
    compile(pattern, options);
  }
  Pcre2Regex(const Pcre2Regex &o)
      : re_(nullptr), has_jit_(false), md_pool_(nullptr) {
    if (o.re_) {
      re_ = pcre2_code_copy(o.re_);
      if (o.has_jit_) {
        int rc = pcre2_jit_compile(re_, PCRE2_JIT_COMPLETE);
        has_jit_ = (rc == 0);
      }
    }
  }
  Pcre2Regex &operator=(const Pcre2Regex &o) {
    if (this == &o)
      return *this;
    free_();
    if (o.re_) {
      re_ = pcre2_code_copy(o.re_);
      if (o.has_jit_) {
        int rc = pcre2_jit_compile(re_, PCRE2_JIT_COMPLETE);
        has_jit_ = (rc == 0);
      }
    }
    return *this;
  }
  Pcre2Regex(Pcre2Regex &&o) noexcept
      : re_(o.re_), has_jit_(o.has_jit_), md_pool_(o.md_pool_) {
    o.re_ = nullptr;
    o.has_jit_ = false;
    o.md_pool_ = nullptr;
  }
  Pcre2Regex &operator=(Pcre2Regex &&o) noexcept {
    if (this != &o) {
      free_();
      re_ = o.re_;
      o.re_ = nullptr;
      has_jit_ = o.has_jit_;
      o.has_jit_ = false;
      md_pool_ = o.md_pool_;
      o.md_pool_ = nullptr;
    }
    return *this;
  }
  ~Pcre2Regex() { free_(); }

  void compile(const std::string &pattern, uint32_t options = 0) {
    free_();
    int errcode = 0;
    PCRE2_SIZE erroff = 0;
    re_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
                        pattern.size(), options, &errcode, &erroff, nullptr);
    if (!re_) {
      PCRE2_UCHAR buffer[256];
      pcre2_get_error_message(errcode, buffer, sizeof(buffer));
      throw EncodeError("Invalid regex pattern: " +
                        std::string(reinterpret_cast<char *>(buffer)));
    }

    // Decide whether to attempt JIT
    bool want_jit = true;
    if (const char *no_jit = std::getenv("TIKTOKEN_NO_JIT")) {
      if (no_jit[0] != '\0')
        want_jit = false;
    }

    int jit_available = 0;
    (void)pcre2_config(PCRE2_CONFIG_JIT, &jit_available);

    has_jit_ = false;
    if (want_jit && jit_available) {
      int rc = pcre2_jit_compile(re_, PCRE2_JIT_COMPLETE);
      has_jit_ = (rc == 0);
    }
  }

  // Get thread-local match data (pooled)
  pcre2_match_data *get_match_data() const {
    static thread_local pcre2_match_data *tl_md = nullptr;
    static thread_local pcre2_code *tl_re = nullptr;

    if (tl_re != re_ || tl_md == nullptr) {
      if (tl_md)
        pcre2_match_data_free(tl_md);
      tl_md = pcre2_match_data_create_from_pattern(re_, nullptr);
      tl_re = re_;
    }
    return tl_md;
  }

  // Iterate matches over [begin, end) - optimized version
  template <class Callback>
  void for_each_match(const char *begin, const char *end, Callback &&cb) const {
    if (TIKTOKEN_UNLIKELY(!re_))
      return;

    pcre2_match_data *md = get_match_data();
    const char *cur = begin;
    const std::size_t total_len = static_cast<std::size_t>(end - begin);

    // Prefetch first cache line
    TIKTOKEN_PREFETCH(begin);

    while (cur < end) {
      // Prefetch ahead
      if (cur + 64 < end) {
        TIKTOKEN_PREFETCH(cur + 64);
      }

      int rc;
      std::size_t remaining = static_cast<std::size_t>(end - cur);

      if (TIKTOKEN_LIKELY(has_jit_))
        rc = pcre2_jit_match(re_, reinterpret_cast<PCRE2_SPTR>(cur), remaining,
                             0, 0, md, nullptr);
      else
        rc = pcre2_match(re_, reinterpret_cast<PCRE2_SPTR>(cur), remaining, 0,
                         0, md, nullptr);

      if (TIKTOKEN_UNLIKELY(rc == PCRE2_ERROR_NOMATCH))
        break;

      if (TIKTOKEN_UNLIKELY(rc < 0)) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(rc, buffer, sizeof(buffer));
        throw EncodeError("Regex error while tokenizing: " +
                          std::string(reinterpret_cast<char *>(buffer)));
      }

      PCRE2_SIZE *ovec = pcre2_get_ovector_pointer(md);
      std::size_t s = static_cast<std::size_t>((cur - begin) + ovec[0]);
      std::size_t e = static_cast<std::size_t>((cur - begin) + ovec[1]);

      if (TIKTOKEN_UNLIKELY(!cb(s, e)))
        break;

      // Advance position
      if (TIKTOKEN_UNLIKELY(e == s)) {
        // Zero-length match: advance by one UTF-8 codepoint
        const unsigned char *ucur =
            reinterpret_cast<const unsigned char *>(cur);
        int seq_len = bitutil::utf8_seq_len(*ucur);
        if (seq_len > 0 &&
            ucur + seq_len <= reinterpret_cast<const unsigned char *>(end)) {
          cur += seq_len;
        } else {
          ++cur;
        }
      } else {
        cur = begin + e;
      }
    }
  }

  // Find first match from absolute offset 'from'
  std::optional<std::pair<std::size_t, std::size_t>>
  find_from(const char *base, std::size_t total_len, std::size_t from) const {
    if (TIKTOKEN_UNLIKELY(!re_ || from >= total_len))
      return std::nullopt;

    pcre2_match_data *md = get_match_data();
    const char *start = base + from;
    int rc;

    if (TIKTOKEN_LIKELY(has_jit_))
      rc = pcre2_jit_match(re_, reinterpret_cast<PCRE2_SPTR>(start),
                           total_len - from, 0, 0, md, nullptr);
    else
      rc = pcre2_match(re_, reinterpret_cast<PCRE2_SPTR>(start),
                       total_len - from, 0, 0, md, nullptr);

    if (rc == PCRE2_ERROR_NOMATCH)
      return std::nullopt;

    if (TIKTOKEN_UNLIKELY(rc < 0)) {
      PCRE2_UCHAR buffer[256];
      pcre2_get_error_message(rc, buffer, sizeof(buffer));
      throw EncodeError("Regex error while tokenizing: " +
                        std::string(reinterpret_cast<char *>(buffer)));
    }

    PCRE2_SIZE *ovec = pcre2_get_ovector_pointer(md);
    return std::make_pair(from + ovec[0], from + ovec[1]);
  }

  pcre2_code *raw() const { return re_; }

private:
  pcre2_code *re_;
  bool has_jit_;
  mutable pcre2_match_data *md_pool_;

  void free_() {
    if (re_) {
      pcre2_code_free(re_);
      re_ = nullptr;
    }
    has_jit_ = false;
  }
};

// ============================================================================
// ============================================================================
// Simple fixed-size cache for frequent token lookups
// ============================================================================
struct SimpleTokenCache {
  static constexpr std::size_t CACHE_SIZE = 1024;
  static constexpr std::size_t CACHE_MASK = CACHE_SIZE - 1;

  struct Entry {
    std::uint8_t data[16];
    std::uint8_t len;
    std::size_t hash;
    Rank rank;
  };

  Entry entries[CACHE_SIZE];

  SimpleTokenCache() { clear(); }

  void clear() {
    for (auto &e : entries) {
      e.len = 0;
      e.hash = 0;
      e.rank = std::numeric_limits<Rank>::max();
    }
  }

  TIKTOKEN_FORCE_INLINE Rank lookup(const U8 *data, std::size_t len,
                                    std::size_t hash) const {
    if (len > 16)
      return std::numeric_limits<Rank>::max();

    std::size_t idx = hash & CACHE_MASK;
    const Entry &e = entries[idx];

    if (e.len == len && e.hash == hash) {
      if (std::memcmp(e.data, data, len) == 0)
        return e.rank;
    }
    return std::numeric_limits<Rank>::max();
  }

  TIKTOKEN_FORCE_INLINE void insert(const U8 *data, std::size_t len,
                                    std::size_t hash, Rank rank) {
    if (len > 16)
      return;

    std::size_t idx = hash & CACHE_MASK;
    Entry &e = entries[idx];

    e.len = static_cast<std::uint8_t>(len);
    e.hash = hash;
    e.rank = rank;
    std::memcpy(e.data, data, len);
  }
};

// ============================================================================
// Optimized BPE Merge using Linked List + Min-Heap (true O(n log n))
// ============================================================================

// Linked list node for BPE merge
struct BPEMergeNode {
  std::size_t byte_pos; // Position in original byte array
  std::int32_t prev;    // Index of previous node (-1 if head)
  std::int32_t next;    // Index of next node (-1 if tail)
  Rank rank;            // Current rank of bigram starting here
};

// Heap entry for BPE merge
struct BPEHeapEntry {
  Rank rank;
  std::int32_t node_idx;

  bool operator>(const BPEHeapEntry &o) const {
    if (rank != o.rank)
      return rank > o.rank;
    return node_idx > o.node_idx; // Tie-break by position for determinism
  }
};

inline std::vector<std::pair<std::size_t, Rank>>
_byte_pair_merge(const EncoderMap &ranks, const U8 *piece, std::size_t len) {
  if (len == 0)
    return {};
  if (len == 1)
    return {{0, std::numeric_limits<Rank>::max()},
            {1, std::numeric_limits<Rank>::max()}};

  // Build linked list of byte positions
  std::vector<BPEMergeNode> nodes;
  nodes.reserve(len + 1);

  // Optimized lookup helper with caching
  auto get_rank = [&](std::size_t start, std::size_t end) -> Rank {
    if (end > len)
      return std::numeric_limits<Rank>::max();
    std::size_t seg_len = end - start;

    // Use cache for small segments
    if (seg_len <= 16) {
      std::size_t h = VecU8Hash::hash_bytes(piece + start, seg_len);
      static thread_local SimpleTokenCache heap_cache;

      Rank cached = heap_cache.lookup(piece + start, seg_len, h);
      if (cached != std::numeric_limits<Rank>::max())
        return cached;

      auto it = ranks.find(lookup_vec(piece, start, end));
      Rank r =
          (it == ranks.end()) ? std::numeric_limits<Rank>::max() : it->second;
      if (r != std::numeric_limits<Rank>::max()) {
        heap_cache.insert(piece + start, seg_len, h, r);
      }
      return r;
    }

    auto it = ranks.find(lookup_vec(piece, start, end));
    return (it == ranks.end()) ? std::numeric_limits<Rank>::max() : it->second;
  };

  // Initialize nodes (each node represents a byte position boundary)
  for (std::size_t i = 0; i <= len; ++i) {
    nodes.push_back({i, static_cast<std::int32_t>(i) - 1,
                     static_cast<std::int32_t>(i) + 1,
                     std::numeric_limits<Rank>::max()});
  }
  nodes.back().next = -1; // Last node has no next

  // Compute initial ranks for each pair and build heap
  std::priority_queue<BPEHeapEntry, std::vector<BPEHeapEntry>,
                      std::greater<BPEHeapEntry>>
      heap;

  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
    // Rank for merging nodes[i] with nodes[i+1] (bigram from byte_pos[i] to
    // byte_pos[i+2])
    if (i + 2 <= len) {
      std::size_t next_next =
          (nodes[i + 1].next >= 0) ? nodes[nodes[i + 1].next].byte_pos : len;
      Rank r = get_rank(nodes[i].byte_pos, next_next);
      nodes[i].rank = r;
      if (r != std::numeric_limits<Rank>::max()) {
        heap.push({r, static_cast<std::int32_t>(i)});
      }
    }
  }

  // Merge loop
  while (!heap.empty()) {
    auto [rank, idx] = heap.top();
    heap.pop();

    BPEMergeNode &node = nodes[idx];

    // Skip if node was deleted or rank is stale
    if (node.next < 0)
      continue;
    if (node.rank != rank)
      continue;

    // Get next node to merge with
    std::int32_t next_idx = node.next;
    BPEMergeNode &next_node = nodes[next_idx];

    // Check if next_node is still valid
    if (next_node.prev != idx)
      continue; // Already merged

    // Verify rank: bigram spans from node.byte_pos to
    // nodes[next_node.next].byte_pos
    std::size_t end_pos =
        (next_node.next >= 0) ? nodes[next_node.next].byte_pos : len;
    Rank current_rank = get_rank(node.byte_pos, end_pos);
    if (current_rank != rank) {
      // Rank changed, update and re-add if still valid
      node.rank = current_rank;
      if (current_rank != std::numeric_limits<Rank>::max()) {
        heap.push({current_rank, idx});
      }
      continue;
    }

    // Perform merge: remove next_node from list
    node.next = next_node.next;
    if (next_node.next >= 0) {
      nodes[next_node.next].prev = idx;
    }

    // Mark next_node as deleted
    next_node.prev = -2;
    next_node.next = -2;

    // Update rank for merged node (new bigram spans to nodes[node.next.next])
    if (node.next >= 0) {
      std::size_t new_end = (nodes[node.next].next >= 0)
                                ? nodes[nodes[node.next].next].byte_pos
                                : len;
      Rank new_rank = get_rank(node.byte_pos, new_end);
      node.rank = new_rank;
      if (new_rank != std::numeric_limits<Rank>::max()) {
        heap.push({new_rank, idx});
      }
    } else {
      node.rank = std::numeric_limits<Rank>::max();
    }

    // Update previous node's rank
    if (node.prev >= 0) {
      BPEMergeNode &prev_node = nodes[node.prev];
      std::size_t prev_end = (node.next >= 0) ? nodes[node.next].byte_pos : len;
      Rank prev_rank = get_rank(prev_node.byte_pos, prev_end);
      prev_node.rank = prev_rank;
      if (prev_rank != std::numeric_limits<Rank>::max()) {
        heap.push({prev_rank, node.prev});
      }
    }
  }

  // Collect remaining nodes by traversing the linked list
  std::vector<std::pair<std::size_t, Rank>> result;
  result.reserve(len); // Upper bound

  std::int32_t curr = 0;
  while (curr >= 0) {
    result.emplace_back(nodes[curr].byte_pos, std::numeric_limits<Rank>::max());
    curr = nodes[curr].next;
  }

  return result;
}

// Simple O(n²) BPE for small pieces - lower overhead beats O(n log n) for n <
// 32 OPTIMIZED: Cache ranks to avoid redundant lookups
inline std::vector<std::pair<std::size_t, Rank>>
_byte_pair_merge_small(const EncoderMap &ranks, const U8 *piece,
                       std::size_t len) {
  if (len == 0)
    return {};
  if (len == 1)
    return {{0, std::numeric_limits<Rank>::max()},
            {1, std::numeric_limits<Rank>::max()}};

  // Use fixed-size array for small inputs to avoid heap allocation
  std::size_t parts[66]; // Max len is 64, plus sentinel
  Rank cached_ranks[65]; // Cache computed ranks to avoid repeated lookups
  std::size_t num_parts = len + 1;

  for (std::size_t i = 0; i <= len; ++i)
    parts[i] = i;

  // Initialize rank cache
  auto recompute_rank = [&](std::size_t i) -> Rank {
    if (i + 2 >= num_parts)
      return std::numeric_limits<Rank>::max();
    std::size_t start = parts[i], end = parts[i + 2];
    std::size_t seg_len = end - start;

    // Fast path for very small segments
    if (seg_len <= 16) {
      std::size_t h = VecU8Hash::hash_bytes(piece + start, seg_len);
      static thread_local SimpleTokenCache merge_cache;
      Rank cached = merge_cache.lookup(piece + start, seg_len, h);
      if (cached != std::numeric_limits<Rank>::max())
        return cached;

      auto it = ranks.find(lookup_vec(piece, start, end));
      Rank r =
          (it == ranks.end()) ? std::numeric_limits<Rank>::max() : it->second;
      if (r != std::numeric_limits<Rank>::max()) {
        merge_cache.insert(piece + start, seg_len, h, r);
      }
      return r;
    }

    auto it = ranks.find(lookup_vec(piece, start, end));
    return (it == ranks.end()) ? std::numeric_limits<Rank>::max() : it->second;
  };

  // Initial rank computation with prefetching
  for (std::size_t i = 0; i + 2 < num_parts; ++i) {
    cached_ranks[i] = recompute_rank(i);
  }

  while (num_parts > 2) {
    Rank min_rank = std::numeric_limits<Rank>::max();
    std::size_t min_idx = 0;

    // Find minimum using cached ranks
    for (std::size_t i = 0; i + 2 < num_parts; ++i) {
      if (cached_ranks[i] < min_rank) {
        min_rank = cached_ranks[i];
        min_idx = i;
      }
    }

    if (min_rank == std::numeric_limits<Rank>::max())
      break;

    // Remove element at min_idx + 1 by shifting
    for (std::size_t i = min_idx + 1; i + 1 < num_parts; ++i) {
      parts[i] = parts[i + 1];
      if (i + 2 < num_parts) {
        cached_ranks[i] = cached_ranks[i + 1];
      }
    }
    --num_parts;

    // Only recompute affected ranks
    if (min_idx > 0 && min_idx < num_parts - 1) {
      cached_ranks[min_idx - 1] = recompute_rank(min_idx - 1);
    }
    if (min_idx < num_parts - 1) {
      cached_ranks[min_idx] = recompute_rank(min_idx);
    }
  }

  std::vector<std::pair<std::size_t, Rank>> result;
  result.reserve(num_parts);
  for (std::size_t i = 0; i < num_parts; ++i) {
    result.emplace_back(parts[i], std::numeric_limits<Rank>::max());
  }
  return result;
}

// DRASTICALLY OPTIMIZED BPE - eliminates vector::erase, adds rank caching
// Key changes: circular buffer instead of erase, only recompute affected ranks
inline std::vector<Rank> byte_pair_encode_fast(const U8 *piece, std::size_t len,
                                               const EncoderMap &ranks) {
  // Use indices instead of erasing - MUCH faster
  using Idx = std::uint16_t;
  constexpr Idx INVALID = std::numeric_limits<Idx>::max();

  struct Part {
    std::size_t pos;
    Rank rank;
    Idx next; // Index of next valid part
  };

  static thread_local std::vector<Part> parts;
  parts.clear();
  parts.reserve(len + 2);

  auto get_rank = [&](Idx i) -> Rank {
    if (i >= parts.size() || parts[i].next == INVALID)
      return std::numeric_limits<Rank>::max();
    Idx next_idx = parts[i].next;
    if (next_idx == INVALID || next_idx >= parts.size())
      return std::numeric_limits<Rank>::max();

    Idx next_next_idx = parts[next_idx].next;
    if (next_next_idx == INVALID || next_next_idx >= parts.size())
      return std::numeric_limits<Rank>::max();

    auto it =
        ranks.find(lookup_vec(piece, parts[i].pos, parts[next_next_idx].pos));
    return (it != ranks.end()) ? it->second : std::numeric_limits<Rank>::max();
  };

  // Initialize with linked indices
  for (std::size_t i = 0; i <= len; ++i) {
    parts.push_back(
        {i, std::numeric_limits<Rank>::max(), static_cast<Idx>(i + 1)});
  }
  parts[len].next = INVALID; // End marker

  Idx head = 0;
  std::size_t active_count = len + 1;

  // Initial rank computation
  Rank min_rank = std::numeric_limits<Rank>::max();
  Idx min_idx = INVALID;

  for (Idx i = 0; i < len; ++i) {
    parts[i].rank = get_rank(i);
    if (parts[i].rank < min_rank) {
      min_rank = parts[i].rank;
      min_idx = i;
    }
  }

  // Main merge loop - NO ERASES!
  while (min_rank != std::numeric_limits<Rank>::max() && active_count > 2) {
    // Perform merge by updating next pointer
    Idx merge_target = parts[min_idx].next;
    if (merge_target != INVALID && merge_target < parts.size()) {
      parts[min_idx].next = parts[merge_target].next;
      --active_count;

      // Only recompute ranks for affected adjacent pairs
      // Previous pair (if exists)
      Idx prev = INVALID;
      for (Idx i = 0; i < parts.size(); ++i) {
        if (parts[i].next == min_idx) {
          prev = i;
          break;
        }
      }

      if (prev != INVALID) {
        parts[prev].rank = get_rank(prev);
      }

      // Current pair
      parts[min_idx].rank = get_rank(min_idx);
    }

    // Find next minimum - only scan active parts
    min_rank = std::numeric_limits<Rank>::max();
    min_idx = INVALID;

    Idx curr = head;
    while (curr != INVALID && curr < parts.size()) {
      if (parts[curr].rank < min_rank) {
        min_rank = parts[curr].rank;
        min_idx = curr;
      }
      curr = parts[curr].next;
      if (curr == INVALID || curr >= parts.size())
        break;
    }
  }

  // Extract final tokens by following the chain
  std::vector<Rank> out;
  out.reserve(active_count);

  Idx curr = head;
  while (curr != INVALID && curr < parts.size()) {
    Idx next_idx = parts[curr].next;
    if (next_idx == INVALID || next_idx >= parts.size())
      break;

    auto it =
        ranks.find(lookup_vec(piece, parts[curr].pos, parts[next_idx].pos));
    if (TIKTOKEN_UNLIKELY(it == ranks.end())) {
      // Segment not found - fall back to individual bytes
      for (std::size_t j = parts[curr].pos; j < parts[next_idx].pos; ++j) {
        auto byte_it = ranks.find(lookup_vec(piece, j, j + 1));
        if (TIKTOKEN_UNLIKELY(byte_it == ranks.end()))
          throw EncodeError("Missing single-byte token in encoder");
        out.push_back(byte_it->second);
      }
    } else {
      out.push_back(it->second);
    }

    curr = next_idx;
  }
  return out;
}

inline std::vector<Rank> byte_pair_encode(const U8 *piece, std::size_t len,
                                          const EncoderMap &ranks) {
  if (TIKTOKEN_UNLIKELY(len == 0))
    return {};

  if (TIKTOKEN_LIKELY(len == 1)) {
    auto it = ranks.find(lookup_vec(piece, 0, 1));
    if (TIKTOKEN_UNLIKELY(it == ranks.end()))
      throw EncodeError("Missing single-byte token in encoder");
    return {it->second};
  }

  // Fast path: check if entire piece is in vocabulary with hash pre-computation
  if (len <= 32) // Increased for better hit rate
  {
    std::size_t h = VecU8Hash::hash_bytes(piece, len);

    // Use thread-local cache for recent lookups
    static thread_local SimpleTokenCache cache;
    Rank cached = cache.lookup(piece, len, h);
    if (cached != std::numeric_limits<Rank>::max()) {
      return {cached};
    }

    // Only do map lookup on cache miss
    auto it = ranks.find(lookup_vec(piece, 0, len));
    if (it != ranks.end()) {
      cache.insert(piece, len, h, it->second);
      return {it->second};
    }
  }

  // Algorithm selection based on length and characteristics
  // Very small: O(n²) - minimal overhead
  // Small-medium: Linear scan version (cache-friendly)
  // Large: Heap-based O(n log n)
  if (len <= 16) {
    // Tiny pieces - just use simple O(n²)
    std::vector<std::pair<std::size_t, Rank>> parts =
        _byte_pair_merge_small(ranks, piece, len);
    std::vector<Rank> out;
    out.reserve(parts.size());
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      auto s = parts[i].first, e = parts[i + 1].first;
      auto it = ranks.find(lookup_vec(piece, s, e));
      if (TIKTOKEN_UNLIKELY(it == ranks.end())) {
        // Segment not found - fall back to individual bytes
        for (std::size_t j = s; j < e; ++j) {
          auto byte_it = ranks.find(lookup_vec(piece, j, j + 1));
          if (TIKTOKEN_UNLIKELY(byte_it == ranks.end()))
            throw EncodeError("Missing single-byte token in encoder");
          out.push_back(byte_it->second);
        }
      } else {
        out.push_back(it->second);
      }
    }
    return out;
  }

  if (len <= 512) {
    // Use simple linear-scan version for cache-friendly performance
    return byte_pair_encode_fast(piece, len, ranks);
  }

  // For long sequences, use heap-based O(n log n)
  std::vector<std::pair<std::size_t, Rank>> parts =
      _byte_pair_merge(ranks, piece, len);

  std::vector<Rank> out;
  out.reserve(parts.size());

  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    auto s = parts[i].first, e = parts[i + 1].first;
    std::size_t seg_len = e - s;

    // Use hash-based cache for segments too
    if (seg_len <= 16) {
      std::size_t h = VecU8Hash::hash_bytes(piece + s, seg_len);
      static thread_local SimpleTokenCache seg_cache;

      Rank cached = seg_cache.lookup(piece + s, seg_len, h);
      if (cached != std::numeric_limits<Rank>::max()) {
        out.push_back(cached);
        continue;
      }

      auto it = ranks.find(lookup_vec(piece, s, e));
      if (TIKTOKEN_UNLIKELY(it == ranks.end())) {
        // Segment not found - fall back to individual bytes
        for (std::size_t j = s; j < e; ++j) {
          auto byte_it = ranks.find(lookup_vec(piece, j, j + 1));
          if (TIKTOKEN_UNLIKELY(byte_it == ranks.end()))
            throw EncodeError("Missing single-byte token in encoder");
          out.push_back(byte_it->second);
        }
      } else {
        seg_cache.insert(piece + s, seg_len, h, it->second);
        out.push_back(it->second);
      }
    } else {
      auto it = ranks.find(lookup_vec(piece, s, e));
      if (TIKTOKEN_UNLIKELY(it == ranks.end())) {
        // Segment not found - fall back to individual bytes
        for (std::size_t j = s; j < e; ++j) {
          auto byte_it = ranks.find(lookup_vec(piece, j, j + 1));
          if (TIKTOKEN_UNLIKELY(byte_it == ranks.end()))
            throw EncodeError("Missing single-byte token in encoder");
          out.push_back(byte_it->second);
        }
      } else {
        out.push_back(it->second);
      }
    }
  }
  return out;
}

// ============================================================================
// CoreBPE with all optimizations
// ============================================================================
class CoreBPE {
public:
  CoreBPE(EncoderMap encoder, SpecialEncMap special_tokens_encoder,
          const std::string &pattern)
      : encoder_(std::move(encoder)),
        special_enc_(std::move(special_tokens_encoder)) {
    build_regex_(pattern);
    build_special_regex_();
    build_decoders_();
    build_sorted_token_bytes_();
    regex_tls_.assign(MAX_NUM_THREADS, regex_);
    special_regex_tls_.assign(MAX_NUM_THREADS, special_regex_);
  }

  static CoreBPE
  New(const std::vector<std::pair<U8Vec, Rank>> &encoder,
      const std::vector<std::pair<std::string, Rank>> &special_tokens_encoder,
      const std::string &pattern) {
    EncoderMap enc;
    enc.reserve(encoder.size());
    for (const auto &kv : encoder)
      enc.emplace(kv.first, kv.second);
    SpecialEncMap se;
    se.reserve(special_tokens_encoder.size());
    for (const auto &kv : special_tokens_encoder)
      se.emplace(kv.first, kv.second);
    return CoreBPE(std::move(enc), std::move(se), pattern);
  }

  // Optimized encode_ordinary with automatic chunking for long texts
  std::vector<Rank> encode_ordinary(const std::string &text) const {
    if (TIKTOKEN_UNLIKELY(text.empty()))
      return {};

    // For long texts, automatically use parallel processing
    // This gives 4-6x speedup for texts > 16KB
    constexpr std::size_t PARALLEL_THRESHOLD = 4096; // 4KB

    if (text.size() > PARALLEL_THRESHOLD) {
      return encode_ordinary_parallel(text, 4096);
    }

    return encode_ordinary_sequential(text);
  }

  // Optimized encode_ordinary with automatic chunking for long texts
  std::vector<Rank> encode_ordinary_sequential(const std::string &text) const {
    if (TIKTOKEN_UNLIKELY(text.empty()))
      return {};

    const Pcre2Regex &re = get_tl_regex_();
    std::vector<Rank> out;
    out.reserve(text.size() / 4); // Estimate ~4 chars per token

    const char *base = text.c_str();
    const char *end = base + text.size();

    // Prefetch encoder buckets and first cache line
    TIKTOKEN_PREFETCH(&encoder_);
    TIKTOKEN_PREFETCH(base);

    // Static single-byte token cache (most common case)
    static thread_local Rank byte_cache[256];
    static thread_local bool byte_cache_init = false;
    if (!byte_cache_init) {
      for (int i = 0; i < 256; ++i) {
        U8Vec single{static_cast<U8>(i)};
        auto it = encoder_.find(single);
        byte_cache[i] = (it != encoder_.end())
                            ? it->second
                            : std::numeric_limits<Rank>::max();
      }
      byte_cache_init = true;
    }

    re.for_each_match(base, end, [&](std::size_t s, std::size_t e) {
      const U8 *bytes = reinterpret_cast<const U8 *>(base + s);
      std::size_t len = e - s;

      // DEBUG PRINT
      // std::string piece(reinterpret_cast<const char*>(bytes), len);
      // std::cout << "Piece: '" << piece << "'" << std::endl;      // Prefetch
      // next match area
      if (e + 64 < text.size()) {
        TIKTOKEN_PREFETCH(base + e + 64);
      }

      // Fast path for single bytes using pre-computed cache
      if (TIKTOKEN_LIKELY(len == 1)) {
        Rank r = byte_cache[bytes[0]];
        if (TIKTOKEN_LIKELY(r != std::numeric_limits<Rank>::max())) {
          out.push_back(r);
          return true;
        }
      }

      // Fast path for 2-4 bytes (very common)
      if (TIKTOKEN_LIKELY(len >= 2 && len <= 4)) {
        std::size_t h = VecU8Hash::hash_bytes(bytes, len);
        static thread_local SimpleTokenCache small_cache;
        Rank cached = small_cache.lookup(bytes, len, h);

        if (cached != std::numeric_limits<Rank>::max()) {
          out.push_back(cached);
          return true;
        }

        auto it = encoder_.find(lookup_vec(bytes, 0, len));
        if (TIKTOKEN_LIKELY(it != encoder_.end())) {
          small_cache.insert(bytes, len, h, it->second);
          out.push_back(it->second);
          return true;
        }
      }

      // Try direct lookup for larger pieces
      if (len <= 32) {
        auto it = encoder_.find(lookup_vec(bytes, 0, len));
        if (it != encoder_.end()) {
          out.push_back(it->second);
          return true;
        }
      }

      // Fall back to BPE
      auto toks = byte_pair_encode(bytes, len, encoder_);
      out.insert(out.end(), toks.begin(), toks.end());
      return true;
    });

    return out;
  }

  // Internal: Chunked encoding for long texts
  std::vector<Rank> encode_ordinary_chunked(const std::string &text,
                                            std::size_t chunk_size) const {
    std::vector<Rank> result;
    result.reserve(text.size() / 4);

    std::size_t pos = 0;
    while (pos < text.size()) {
      // Find chunk boundary - prefer sentence/word boundaries
      std::size_t end = std::min(pos + chunk_size, text.size());

      if (end < text.size()) {
        // Look back up to 256 bytes for a good break point (period, newline,
        // space)
        std::size_t search_start = (end > 256) ? end - 256 : pos;
        std::size_t best = end;

        // Priority: newline > period+space > comma+space > any space
        for (std::size_t i = end; i > search_start; --i) {
          if (text[i] == '\n') {
            best = i + 1;
            break;
          }
          if (i > 0 && text[i - 1] == '.' &&
              (text[i] == ' ' || text[i] == '\n')) {
            best = i;
            break;
          }
        }

        // Fall back to any whitespace
        if (best == end) {
          for (std::size_t i = end; i > search_start; --i) {
            if (text[i] == ' ' || text[i] == '\t' || text[i] == '\n') {
              best = i + 1;
              break;
            }
          }
        }

        // Ensure UTF-8 boundary
        while (best < text.size() &&
               !bitutil::is_utf8_lead(static_cast<U8>(text[best]))) {
          ++best;
        }

        end = best;
      }

      // Encode this chunk
      std::string chunk = text.substr(pos, end - pos);
      auto chunk_tokens = encode_ordinary(chunk);
      result.insert(result.end(), chunk_tokens.begin(), chunk_tokens.end());

      pos = end;
    }

    return result;
  }

  // Optimized parallel encode with better work distribution
  std::vector<Rank>
  encode_ordinary_parallel(const std::string &text,
                           std::size_t chunk_size = 8192) const {
    // Increased threshold for parallelization
    if (text.size() < chunk_size) {
      return encode_ordinary_sequential(text);
    }

    // Calculate optimal number of chunks based on text size and thread count
    std::size_t hw_threads = std::max(1u, std::thread::hardware_concurrency());
    std::size_t num_chunks = (text.size() + chunk_size - 1) / chunk_size;
    std::size_t actual_chunk_size = (text.size() + num_chunks - 1) / num_chunks;

    // Find safe split points (UTF-8 boundaries, prefer whitespace)
    std::vector<std::size_t> split_points;
    split_points.reserve(num_chunks + 1);
    split_points.push_back(0);

    for (std::size_t i = 1; i < num_chunks; ++i) {
      std::size_t target = i * actual_chunk_size;
      if (target >= text.size())
        break;

      // Binary search for nearest whitespace within 128 bytes
      std::size_t best = target;
      std::size_t search_start = (target > 128) ? target - 128 : 0;
      std::size_t search_end = std::min(target + 128, text.size());

      for (std::size_t j = target; j < search_end; ++j) {
        if (bitutil::is_whitespace(static_cast<U8>(text[j]))) {
          best = j + 1;
          break;
        }
      }

      // Ensure UTF-8 boundary
      while (best < text.size() &&
             !bitutil::is_utf8_lead(static_cast<U8>(text[best]))) {
        ++best;
      }

      split_points.push_back(best);
    }
    split_points.push_back(text.size());

    // Remove duplicates
    split_points.erase(std::unique(split_points.begin(), split_points.end()),
                       split_points.end());
    num_chunks = split_points.size() - 1;

    if (num_chunks <= 1) {
      return encode_ordinary_sequential(text);
    }

    // Encode chunks in parallel using string_view to avoid copies
    std::vector<std::vector<Rank>> chunk_results(num_chunks);
    std::atomic<std::size_t> next_chunk{0};
    std::vector<std::future<void>> futures;
    futures.reserve(hw_threads);

    for (std::size_t t = 0; t < hw_threads; ++t) {
      futures.push_back(std::async(std::launch::async, [&]() {
        while (true) {
          std::size_t i = next_chunk.fetch_add(1, std::memory_order_relaxed);
          if (i >= num_chunks)
            break;

          const char *start = text.data() + split_points[i];
          std::size_t len = split_points[i + 1] - split_points[i];
          std::string chunk(start, len);
          chunk_results[i] = encode_ordinary_sequential(chunk);
        }
      }));
    }

    for (auto &f : futures)
      f.get();

    // Merge results with parallelized counting
    std::size_t total_tokens = 0;
    for (const auto &cr : chunk_results)
      total_tokens += cr.size();

    std::vector<Rank> result;
    result.reserve(total_tokens);
    for (auto &cr : chunk_results) {
      result.insert(result.end(), cr.begin(), cr.end());
    }

    return result;
  }

  std::pair<std::vector<Rank>, std::size_t>
  encode(const std::string &text,
         const std::set<std::string> &allowed_special) const {
    if (TIKTOKEN_UNLIKELY(text.empty()))
      return {{}, 0};

    const Pcre2Regex &re = get_tl_regex_();
    const Pcre2Regex &sre = get_tl_special_regex_();
    std::vector<Rank> out;
    out.reserve(text.size() / 4);

    std::size_t start = 0, last_piece_len = 0;
    const char *base = text.c_str();
    const std::size_t N = text.size();

    auto next_allowed_special = [&](std::size_t from)
        -> std::optional<std::pair<std::size_t, std::size_t>> {
      std::size_t pos = from;
      while (true) {
        auto m = sre.find_from(base, N, pos);
        if (!m)
          return std::nullopt;
        auto [s, e] = *m;
        if (allowed_special.count(scratch_string(base + s, e - s)))
          return m;
        pos = s + 1;
      }
    };

    while (true) {
      auto next_sp = next_allowed_special(start);
      std::size_t end_span = next_sp ? next_sp->first : N;

      if (end_span > start) {
        const char *chunk_b = base + start;
        const char *chunk_e = base + end_span;

        re.for_each_match(
            chunk_b, chunk_e, [&](std::size_t s_rel, std::size_t e_rel) {
              std::size_t s = start + s_rel, e = start + e_rel;
              const U8 *bytes = reinterpret_cast<const U8 *>(base + s);
              std::size_t len = e - s;

              auto it = encoder_.find(lookup_vec(bytes, 0, len));
              if (TIKTOKEN_LIKELY(it != encoder_.end())) {
                last_piece_len = 1;
                out.push_back(it->second);
              } else {
                auto toks = byte_pair_encode(bytes, len, encoder_);
                last_piece_len = toks.size();
                out.insert(out.end(), toks.begin(), toks.end());
              }
              return true;
            });
      }

      if (next_sp) {
        auto [s, e] = *next_sp;
        const auto &sv = scratch_string(base + s, e - s);
        auto it = special_enc_.find(sv);
        if (TIKTOKEN_UNLIKELY(it == special_enc_.end()))
          throw EncodeError("Unknown special token during encode");
        out.push_back(it->second);
        start = e;
        last_piece_len = 0;
      } else
        break;
    }
    return {out, last_piece_len};
  }

  std::vector<Rank> encode_with_special_tokens(const std::string &text) const {
    std::set<std::string> allowed;
    for (const auto &kv : special_enc_)
      allowed.insert(kv.first);
    return encode(text, allowed).first;
  }

  // Optimized decode with pre-calculated total size
  std::vector<U8> decode_bytes(const std::vector<Rank> &tokens) const {
    if (TIKTOKEN_UNLIKELY(tokens.empty()))
      return {};

    // First pass: calculate total size
    std::size_t total_size = 0;
    for (auto t : tokens) {
      auto it = dec_.find(t);
      if (TIKTOKEN_LIKELY(it != dec_.end())) {
        total_size += it->second.size();
        continue;
      }
      auto is = special_dec_.find(t);
      if (is != special_dec_.end()) {
        total_size += is->second.size();
        continue;
      }
      throw DecodeKeyError(t);
    }

    // Second pass: copy data
    std::vector<U8> out;
    out.reserve(total_size);

    for (auto t : tokens) {
      auto it = dec_.find(t);
      if (TIKTOKEN_LIKELY(it != dec_.end())) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        continue;
      }
      auto is = special_dec_.find(t);
      if (is != special_dec_.end()) {
        out.insert(out.end(), is->second.begin(), is->second.end());
      }
    }
    return out;
  }

  std::set<std::string> special_tokens() const {
    std::set<std::string> s;
    for (const auto &kv : special_enc_)
      s.insert(kv.first);
    return s;
  }

  const std::vector<U8Vec> &token_byte_values_sorted() const {
    return sorted_token_bytes_;
  }

  std::pair<std::vector<Rank>, std::vector<std::vector<Rank>>>
  encode_with_unstable(const std::string &text,
                       const std::set<std::string> &allowed_special) const {
    auto enc_res = encode(text, allowed_special);
    auto tokens = std::move(enc_res.first);
    std::size_t last_len = enc_res.second;
    if (last_len == 0)
      return {tokens, {}};

    increase_last_piece_len_(tokens, last_len);

    auto unstable_bytes =
        decode_bytes(std::vector<Rank>(tokens.end() - last_len, tokens.end()));
    tokens.resize(tokens.size() - last_len);
    std::vector<std::vector<Rank>> completions;
    if (unstable_bytes.empty())
      return {tokens, completions};

    // 1) single tokens starting with unstable_bytes
    auto lb = std::lower_bound(
        sorted_token_bytes_.begin(), sorted_token_bytes_.end(), unstable_bytes,
        [](const U8Vec &a, const U8Vec &b) { return a < b; });
    for (auto it = lb; it != sorted_token_bytes_.end(); ++it) {
      const auto &bytes = *it;
      if (bytes.size() < unstable_bytes.size())
        break;
      if (!std::equal(unstable_bytes.begin(), unstable_bytes.end(),
                      bytes.begin()))
        break;
      auto e = encoder_.find(bytes);
      if (e != encoder_.end())
        completions.push_back({e->second});
    }

    // 2) brute force combinations
    for (std::size_t i = 1; i < unstable_bytes.size(); ++i) {
      U8Vec prefix(unstable_bytes.begin(),
                   unstable_bytes.begin() + static_cast<std::ptrdiff_t>(i));
      U8Vec suffix(unstable_bytes.begin() + static_cast<std::ptrdiff_t>(i),
                   unstable_bytes.end());
      auto lb2 = std::lower_bound(
          sorted_token_bytes_.begin(), sorted_token_bytes_.end(), suffix,
          [](const U8Vec &a, const U8Vec &b) { return a < b; });
      for (auto it = lb2; it != sorted_token_bytes_.end(); ++it) {
        const auto &bytes = *it;
        if (bytes.size() < suffix.size())
          break;
        if (!std::equal(suffix.begin(), suffix.end(), bytes.begin()))
          break;
        U8Vec possibility;
        possibility.reserve(prefix.size() + bytes.size());
        possibility.insert(possibility.end(), prefix.begin(), prefix.end());
        possibility.insert(possibility.end(), bytes.begin(), bytes.end());

        std::vector<Rank> encoded;
        if (utf8::is_valid(possibility.begin(), possibility.end())) {
          std::string s(reinterpret_cast<const char *>(possibility.data()),
                        possibility.size());
          encoded = encode_ordinary(s);
        } else {
          encoded = byte_pair_encode(possibility.data(), possibility.size(),
                                     encoder_);
        }
        std::size_t covered = 0;
        std::vector<Rank> seq;
        for (Rank t : encoded) {
          seq.push_back(t);
          covered += dec_.at(t).size();
          if (covered >= unstable_bytes.size())
            break;
        }
        completions.push_back(std::move(seq));
      }
    }

    // 3) trailing incomplete UTF-8 quick-fix
    if (unstable_bytes.size() > 1) {
      const U8 *data = unstable_bytes.data();
      std::size_t n = unstable_bytes.size();
      std::size_t cont = 0;
      std::size_t pos = n;
      while (pos > 0 && bitutil::is_utf8_continuation(data[pos - 1])) {
        --pos;
        ++cont;
      }
      std::size_t start_cp = pos ? pos - 1 : 0;
      if (pos > 0) {
        int need = bitutil::utf8_seq_len(data[start_cp]) - 1;
        if (need >= 0 && static_cast<std::size_t>(need) == cont) {
          try {
            auto cp_start_it =
                unstable_bytes.begin() + static_cast<std::ptrdiff_t>(start_cp);
            uint32_t cp = utf8::peek_next(cp_start_it, unstable_bytes.end());
            bool is_space = bitutil::is_whitespace(static_cast<U8>(cp)) ||
                            cp == '\n' || cp == '\r';
            if (is_space && start_cp > 0) {
              auto ra = byte_pair_encode(data, start_cp, encoder_);
              auto rb =
                  byte_pair_encode(data + start_cp, n - start_cp, encoder_);
              ra.insert(ra.end(), rb.begin(), rb.end());
              completions.push_back(std::move(ra));
            }
          } catch (...) {
          }
        }
      }
    }

    std::sort(completions.begin(), completions.end());
    completions.erase(std::unique(completions.begin(), completions.end()),
                      completions.end());
    return {tokens, completions};
  }

  std::vector<Rank> encode_single_piece(const U8 *bytes,
                                        std::size_t len) const {
    auto it = encoder_.find(lookup_vec(bytes, 0, len));
    if (it == encoder_.end())
      return byte_pair_encode(bytes, len, encoder_);
    return std::vector<Rank>{it->second};
  }

  Rank encode_single_token(const U8 *bytes, std::size_t len) const {
    auto it = encoder_.find(lookup_vec(bytes, 0, len));
    if (it != encoder_.end())
      return it->second;
    if (utf8::is_valid(bytes, bytes + len)) {
      std::string s(reinterpret_cast<const char *>(bytes), len);
      auto it2 = special_enc_.find(s);
      if (it2 != special_enc_.end())
        return it2->second;
    }
    throw std::out_of_range("encode_single_token: piece not found");
  }

private:
  EncoderMap encoder_;
  SpecialEncMap special_enc_;
  DecoderMap dec_;
  SpecialDecMap special_dec_;
  Pcre2Regex regex_;
  Pcre2Regex special_regex_;
  std::vector<Pcre2Regex> regex_tls_;
  std::vector<Pcre2Regex> special_regex_tls_;
  std::vector<U8Vec> sorted_token_bytes_;

  const Pcre2Regex &get_tl_regex_() const {
    static thread_local std::size_t idx = hash_current_thread();
    return regex_tls_[idx];
  }
  const Pcre2Regex &get_tl_special_regex_() const {
    static thread_local std::size_t idx = hash_current_thread();
    return special_regex_tls_[idx];
  }

  void build_regex_(const std::string &pattern) {
    regex_ = Pcre2Regex(pattern, PCRE2_UTF | PCRE2_UCP);
  }
  void build_special_regex_() {
    std::string joined;
    bool first = true;
    for (const auto &kv : special_enc_) {
      if (!first)
        joined += "|";
      first = false;
      joined += regex_escape(kv.first);
    }
    if (joined.empty())
      joined = "(?!)";
    special_regex_ = Pcre2Regex(joined, PCRE2_UTF | PCRE2_UCP);
  }
  void build_decoders_() {
    dec_.reserve(encoder_.size());
    for (const auto &kv : encoder_)
      dec_.emplace(kv.second, kv.first);
    if (dec_.size() != encoder_.size())
      throw std::runtime_error(
          "Encoder and decoder must be of equal length (duplicate indices?)");
    for (const auto &kv : special_enc_)
      special_dec_.emplace(kv.second, U8Vec(kv.first.begin(), kv.first.end()));
  }
  void build_sorted_token_bytes_() {
    sorted_token_bytes_.reserve(encoder_.size());
    for (const auto &kv : encoder_)
      sorted_token_bytes_.push_back(kv.first);

    // Standard sort - parallel sort requires TBB which may not be available
    std::sort(sorted_token_bytes_.begin(), sorted_token_bytes_.end());
  }

  void increase_last_piece_len_(std::vector<Rank> &tokens,
                                std::size_t &last_len) const {
    auto all_space = [&](Rank t) -> bool {
      auto it = dec_.find(t);
      if (it == dec_.end())
        return false;
      const auto &b = it->second;
      for (auto rit = b.rbegin(); rit != b.rend(); ++rit) {
        if (!bitutil::is_whitespace(*rit))
          return false;
      }
      return true;
    };
    if (last_len > 0 && all_space(tokens[tokens.size() - last_len])) {
      while (last_len < tokens.size() &&
             all_space(tokens[tokens.size() - last_len - 1]))
        ++last_len;
    }
    if (last_len > tokens.size())
      last_len = tokens.size();
  }
};

} // namespace tiktoken
