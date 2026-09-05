# LargeLanguageModelcpp

A C++ implementation of a GPT-like large language model from scratch, built with **LibTorch** (PyTorch C++ API). This project is a learning exercise to understand transformer architecture, attention mechanisms (including modern variants), tokenization, KV caching, data pipelines, and full pre-training loops—all in C++.

## Features

### Core Transformer Architecture
- **Attention Mechanism** : Self-attention with learned Q, K, V weight matrices (scaled dot-product attention)
- **Multi-Head Causal Attention** : Multi-head self-attention with causal masking (GPT-2 style)
- **KV Cache** : Key-value caching for fast iterative text generation (`MultiHeadAttentionV2`, `GptModelV2`)
- **Transformer Blocks** : Full GPT-2 style block (LayerNorm → Attention → shortcut, LayerNorm → FeedForward → shortcut)
- **GELU Activation** : GPT-2 style GELU activation in the feed-forward network
- **Text Generation** : Greedy text generation with and without KV cache (`generate_text_simple` / `generateTextSimpleCached`)
- **GPT Dataset V1** : Custom dataset loader for text-based training data (`the-verdict.txt`)
- **Tokenizer** : OpenAI's **tiktoken** C++ port for BPE (Byte-Pair Encoding) tokenization
- **Embeddings** : Token + position embedding layer

### Modern Attention Variants
- **Group Query Attention (GQA)** : `GroupQueryAttentionV1` — query heads split into `n_kv_groups`; each group of `groupSize = numHeads / numKvGroups` query heads shares a single K/V head. Reduces KV-cache memory footprint and attention bandwidth during decoding at a small accuracy cost vs full MHA.
- **Multi-Head Latent Attention (MLA)** : `MultiHeadLatentV1` / `gpt2mhl` — latent (low-rank) compression of the K/V projections (DeepSeek-style) to shrink KV-cache memory even further than GQA.

### Pre-Training Pipeline
- **Data Preparation** : `PreparedData` (`dataPreparation.hpp/cpp`) — tokenizer loading, text↔token-ID conversion, batch encoding, and sliding-window GPT dataset construction.
- **DataLoader** : Custom `createDataLoaderV2` with `SequentialSampler` matching Python's `DataLoader(shuffle=False)`, producing `[batch, seq_len]` input/target batches (targets = inputs shifted by 1).
- **Cross-Entropy Loss** : `cal_loss_batch` / `total_loss_loader` — batch loss and full-loader loss evaluation (mirrors PyTorch's `calc_loss_batch` / `calc_loss_loader`).
- **Training Loop** : `train_model_simple` — AdamW optimizer (`lr=4e-4`, `weight_decay=0.1`), epoch/step tracking, periodic train/val loss evaluation, CUDA cache clearing, and text-sample generation during training.
- **Sampling** : `generate_with_temperature` — top-k filtering + temperature-scaled softmax + `multinomial` sampling, with optional EOS-token early stopping.

### Analysis Tooling
- **FLOPs Analysis** : `FlopsAnalysis.cpp` — estimates floating-point operations for attention layers.
- **Memory Estimation** : `memoryEstimateMHA_GQA.cpp` — compares KV-cache memory footprint of full Multi-Head Attention vs Group Query Attention.
- **Unit Tests** : **doctest** — all components are tested through `TEST_CASE` macros (no `main()`; doctest provides the entry point).

## Architecture

The model mirrors the GPT-2 architecture (configurable via `config` struct, default = GPT-2 124M):

```
Input Tokens [batch, seq_len]
    │
    ▼
┌─────────────────────────────────┐
│  Token Embedding + Position     │
│  Embedding + Dropout            │
└─────────────────────────────────┘
    │
    ▼  (repeated n_layers times)
┌─────────────────────────────────┐
│  Transformer Block              │
│  ┌───────────────────────────┐  │
│  │ LayerNorm1                │  │
│  │ Multi-Head Causal Attn    │  │  ← MHA / GQA / MLA variants
│  │ + shortcut                │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │ LayerNorm2                │  │
│  │ GELU Feed-Forward         │  │
│  │ + shortcut                │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Final LayerNorm                │
│  Output Head [vocab_size]       │
└─────────────────────────────────┘
```

## Module Layout

### KV-Cache Implementation — `include/Gpt2Untrained/Tests/kv_cache/`

| Module | File | Purpose |
|---|---|---|
| `GptModelV2` | `GptModelV2.hpp/cpp` | Full GPT-2 style model with token+position embedding, stacked transformer blocks, final norm, output head, and KV-cache lifecycle (`reset_kv_cache`) |
| `TransformBlockV2` | `TransformBlockV2.hpp/cpp` | One transformer block: LayerNorm→Attention→dropout→shortcut, then LayerNorm→FeedForward→dropout→shortcut |
| `MultiHeadAttentionV2` | `MultiHeadAttentionV2.hpp/cpp` | Multi-head causal self-attention with an optional sliding-window KV cache (`cache_k`, `cache_v`, `current_ptr`) |
| `FeedForward` | `FeedForward.hpp/cpp` | Two-layer MLP (`emb_dim` → `4×emb_dim` → `emb_dim`) with GELU |
| `LayerNormV2` | `LayerNormV2.hpp/cpp` | GPT-2 style learnable LayerNorm with affine parameters |
| `GELU` | `GELU.hpp/cpp` | GPT-2 style GELU activation (tanh approximation) |

### Group Query Attention — `include/Gpt2Untrained/Tests/GroupQueryAttention/`

| Module | File | Purpose |
|---|---|---|
| `GroupQueryAttentionV1` | `GroupQueryAttentionV1.hpp/cpp` | GQA attention layer; projects Q to `numHeads×head_dim` but K/V to `numKvGroups×head_dim`, then shares each K/V head across a group of query heads via `repeat_interleave`. Maintains a growing KV cache. |
| `Gpt2ModelV3` | `Gpt2ModelV3.hpp/cpp` | GPT-2 model variant using GQA attention with KV cache lifecycle |
| `TransformerBlockV4` | `TransformerBlockV4.hpp/cpp` | Transformer block with GQA attention |
| `FeedForwardV3` | `FeedForwardV3.hpp/cpp` | GELU feed-forward network |
| `LayerNormV3` | `LayerNormV3.hpp/cpp` | Learnable LayerNorm |
| `GELUV3` | `GELUV3.hpp/cpp` | GPT-2 style GELU |

### Multi-Head Latent Attention — `include/Gpt2Untrained/Tests/MultiHeadLatent/`

| Module | File | Purpose |
|---|---|---|
| `gpt2mhl` | `gpt2mhl.hpp/cpp` | End-to-end GPT-2 model with Multi-Head Latent Attention; supports KV cache, loss, generation (`generateTextSimpleCachedV2`), and pre-training |
| `MultiHeadLatentV1` | `MultiHeadLatentV1.hpp/cpp` | Latent (low-rank) K/V compression attention layer to minimize KV-cache memory |
| `TransformerBlockV5` | `TransformerBlockV5.hpp/cpp` | Transformer block using the latent attention layer |

### Data Preparation & Pre-Training

| Module | File | Purpose |
|---|---|---|
| `PreparedData` | `include/Gpt2Untrained/dataPreparation.hpp`, `src/Gpt2Untrained/dataPreparation.cpp` | Tokenizer loading, text↔token-ID conversion, batch encoding |
| `GPTDatasetV1` | `include/basics/GPTDatasetV1.h` | Sliding-window dataset producing input/target pairs (target = input shifted by 1) |
| `preTrainingOnunlabled.cpp` | `src/TrainedGpt/preTrainingOnunlabled.cpp` | Full pre-training driver: data loaders, cross-entropy loss, AdamW training loop, loss evaluation, temperature/top-k sampling |

## KV-Cache Text Generation

The test `kv_caching` in `kv_cache.cpp` implements a C++ port of `generate_text_simple_cached`:

```cpp
torch::Tensor generateTextSimpleCached(
    GptModelV2& model,
    torch::Tensor idx,
    int max_new_tokens,
    int context_size = 0,
    bool use_cache = true
);
```

**How it works:**

1. **Prefill phase** — The input prompt is processed in chunks of `kv_window_size`, filling the cache incrementally (handles prompts longer than the window).
2. **max_generable cap** — The number of new tokens is limited to `ctx_len - input_length` because of the fixed-size (learned) position embedding.
3. **Cached decode loop** — Each step feeds only the single newly-predicted token `[batch, 1]` into the model with `use_cache=true`. The attention layers re-read the growing `cache_k`/`cache_v` instead of recomputing keys/values for the entire context — this is the standard optimization used by inference engines like llama.cpp and vLLM.

The KV cache itself (in `MultiHeadAttentionV2`) maintains:
- `cache_k`, `cache_v` — buffers of shape `[batch, heads, window_size, head_dim]`
- `current_ptr` — how many valid slots are filled
- **Overflow handling** — when a new chunk doesn't fit, the oldest tokens are discarded (sliding-window behavior)
- **Offset causal mask** — when serving a partial attention matrix (cached path), the mask is offset by `K - num_tokens` so newer query rows can attend to all past (cached) keys correctly

## Attention Variants Comparison

| Variant | KV Heads | KV-cache memory | Notes |
|---|---|---|---|
| Multi-Head Attention (MHA) | `numHeads` | `numHeads × head_dim` per token | Baseline, full expressiveness |
| Group Query Attention (GQA) | `numKvGroups` | `numKvGroups × head_dim` per token | Greatly reduced cache; small accuracy cost. Groups of query heads share one K/V head. |
| Multi-Head Latent Attention (MLA) | latent (compressed) | low-rank compressed per token | Even smaller cache than GQA via latent K/V compression |

## Pre-Training

Pre-training is driven from `src/TrainedGpt/preTrainingOnunlabled.cpp` using the `gpt2mhl` (Multi-Head Latent Attention) model. The pipeline mirrors the classic GPT-2 "from scratch" training recipe:

1. **Data split** — the corpus (e.g. `the-verdict.txt`) is split into train/validation sets (configurable ratio, e.g. 90/10 or 70/30).
2. **DataLoaders** — `createDataLoaderV2` builds `[batch, seq_len]` batches with a sliding-window stride; targets are inputs shifted by one token.
3. **Loss** — `cal_loss_batch` computes cross-entropy on the flattened logits vs targets; `total_loss_loader` averages over a loader.
4. **Optimizer** — AdamW (`lr = 4e-4`, `weight_decay = 0.1`); gradients zeroed with `set_to_none=true` to free memory; CUDA cache emptied periodically.
5. **Evaluation** — every `eval_freq` steps the model is evaluated on both train and validation loaders, and a sample is generated to visually track progress.

### Sampling

`generate_with_temperature` supports:
- Greedy decoding (default)
- **Top-k filtering** — keep only the `top_k` most likely logits before softmax
- **Temperature scaling** — `logits / temperature` before softmax for diversity control
- **EOS early stop** — stops generation when an optional end-of-sequence token is sampled

## Pretrained Model Weights

The randomly-initialized model must be pre-trained before it produces coherent text. You can download the **GPT-2 small (124M)** pretrained weights (PyTorch format) trained from scratch:

```
https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/gpt2-small-124M.pth
```

Save it as `models/gpt2-small-124M.pth` (or your preferred path) and load it into the model with `torch::load` before inference/training:

```cpp
torch::load(model, "models/gpt2-small-124M.pth");
```

> **Note:** These weights are in PyTorch `.pth` format. When loading into the C++ model, ensure the state-dict keys and shapes match the architecture (MHA vs GQA vs MLA variants have different KV projection shapes).

## Prerequisites

| Dependency | Version / Notes |
|---|---|
| **CMake** | ≥ 3.20 |
| **C++ Compiler** | GCC ≥ 10, Clang ≥ 12 (C++20) |
| **LibTorch** | 2.x (download from [pytorch.org](https://pytorch.org/get-started/locally/)) |
| **vcpkg** | For `fmt`, `pcre2`, `curl` |
| **CUDA Toolkit** | ≥ 11.x (for GPU acceleration) |
| **libutfcpp-dev** | UTF-8 library for tiktoken (`sudo apt install libutfcpp-dev`) |
| **libpcre2-dev** | Regex library for tiktoken (`sudo apt install libpcre2-dev`) |

## Setting Up LibTorch

LibTorch is **excluded from git** (see `.gitignore`) because it's several gigabytes. You need to download it manually:

```bash
# 1. Go to https://pytorch.org/get-started/locally/
#    Select: Linux → LibTorch → C++ / Java → (stable / CUDA)

# 2. Download and extract into external/
cd external/
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
# Now external/libtorch/ exists

# 3. Verify the directory structure:
ls external/libtorch/
# → include/  lib/  share/
```

## Building

### Using CMake with vcpkg (Recommended)

```bash
# Install vcpkg first if you haven't already
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh

# Install required packages
~/vcpkg/vcpkg install fmt pcre2 curl

# Build the project
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake
make -j$(nproc)
```

### Using CLion

1. Open the project in CLion
2. Go to **Settings → Build, Execution, Deployment → CMake**
3. Set **CMake options**: `-DCMAKE_TOOLCHAIN_FILE=/home/youruser/vcpkg/scripts/buildsystems/vcpkg.cmake`
4. Build (Ctrl+F9) and run (Shift+F10)

## How the tiktoken Issue Was Solved

The C++ port of OpenAI's tiktoken ([tikteken.cpp](https://github.com/ztong/tiktoken.cpp)) relies on two system libraries that are not managed by CMake's `find_package` by default:

### 1. **PCRE2** (Perl Compatible Regular Expressions)

Tiktoken uses PCRE2 for regex-based token splitting (e.g., `gpt2` / `cl100k` regex patterns).

**The Problem:** The original CMakeLists tried `find_package(PCRE2 CONFIG)` which fails unless PCRE2 was built with CMake exports.

**The Fix:** Use `pkg-config` to locate PCRE2:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

if(NOT TARGET PCRE2::8BIT)
    add_library(PCRE2::8BIT INTERFACE IMPORTED)
    target_include_directories(PCRE2::8BIT INTERFACE ${PCRE2_INCLUDE_DIRS})
    target_link_libraries(PCRE2::8BIT INTERFACE ${PCRE2_LIBRARIES})
endif()
```

Install PCRE2 with:
```bash
sudo apt install libpcre2-dev
# or with vcpkg: vcpkg install pcre2
```

### 2. **utf8cpp** (UTF-8 C++ utilities)

Tiktoken needs UTF-8 decoding/encoding for handling Unicode characters in tokenized text.

**The Problem:** `find_package(utf8cpp)` fails because different distros package it under different CMake target names.

**The Fix:** Manually create an imported target pointing to the system include path:

```cmake
if(NOT TARGET utf8cpp::utf8cpp)
    add_library(utf8cpp::utf8cpp INTERFACE IMPORTED)
    target_include_directories(utf8cpp::utf8cpp INTERFACE /usr/include/utf8cpp)
endif()
```

Install utf8cpp with:
```bash
sudo apt install libutfcpp-dev
```

### 3. **tiktoken as a Git Submodule**

The tiktoken C++ library is included as a git submodule so its version is pinned:

```bash
# When cloning for the first time:
git clone --recurse-submodules https://github.com/Moin2002-tech/LargeLanguageModelcpp.git

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive
```

The top-level `CMakeLists.txt` adds it with:
```cmake
add_subdirectory(external/tiktoken)
target_link_libraries(LargeLanguageModelcpp PRIVATE tiktoken)
```

### Self-Attention Forward Pass

1. **Linear projections:** `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v`
2. **Score computation:** `scores = Q @ K^T`
3. **Scale & Softmax:** `weights = softmax(scores / sqrt(d_out), dim=1)`
4. **Context vector:** `output = weights @ V`

## Running

```bash
cd build && make
./LargeLanguageModelcpp
```

This runs the doctest test suite which prints attention scores, softmax outputs, context vectors, the full 6×6 attention score matrix, and the pre-training/loss-evaluation diagnostics.

## License

MIT — feel free to use, modify, and learn from this code.