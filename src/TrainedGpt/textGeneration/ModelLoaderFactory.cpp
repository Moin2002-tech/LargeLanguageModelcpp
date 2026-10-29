//
// Created by moinshaikh on 10/29/26.
//
#include<doctest.hpp>
#include <TrainedGpt/textGeneration/ModelLoaderFactory.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), {});
}

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    uint64_t varint() {
        uint64_t r = 0; int shift = 0;
        while (p < end) {
            uint8_t b = *p++;
            r |= uint64_t(b & 0x7f) << shift;
            if (!(b & 0x80)) return r;
            shift += 7;
        }
        throw std::runtime_error("bad varint");
    }
};

Reader make_reader(const std::string& s) {
    return Reader{(const uint8_t*)s.data(), (const uint8_t*)s.data() + s.size()};
}

void skip_field(Reader& r, int wt) {
    if (wt == 0) r.varint();
    else if (wt == 1) r.p += 8;
    else if (wt == 2) { uint64_t l = r.varint(); r.p += l; }
    else if (wt == 5) r.p += 4;
    else throw std::runtime_error("bad wire type");
}

struct Entry {
    int dtype = 0;                  // 1 = DT_FLOAT
    std::vector<int64_t> shape;
    int shard = 0;
    uint64_t offset = 0, size = 0;
};

Entry parse_entry(const std::string& v) {
    Entry e;
    Reader r = make_reader(v);
    while (r.p < r.end) {
        uint64_t tag = r.varint();
        int field = tag >> 3, wt = tag & 7;
        if (wt == 0) {
            uint64_t x = r.varint();
            if (field == 1) e.dtype = (int)x;
            else if (field == 3) e.shard = (int)x;
            else if (field == 4) e.offset = x;
            else if (field == 5) e.size = x;
        } else if (wt == 2) {
            uint64_t len = r.varint();
            const uint8_t* s = r.p; r.p += len;
            if (field == 2) {                               // TensorShapeProto
                Reader sr{s, s + len};
                while (sr.p < sr.end) {
                    uint64_t t = sr.varint();
                    if ((t >> 3) == 2 && (t & 7) == 2) {    // Dim
                        uint64_t l = sr.varint();
                        Reader dr{sr.p, sr.p + l}; sr.p += l;
                        while (dr.p < dr.end) {
                            uint64_t dt = dr.varint();
                            if ((dt >> 3) == 1 && (dt & 7) == 0)
                                e.shape.push_back((int64_t)dr.varint());
                            else skip_field(dr, dt & 7);
                        }
                    } else skip_field(sr, t & 7);
                }
            }
        } else skip_field(r, wt);
    }
    return e;
}

std::string read_block(const std::string& f, uint64_t off, uint64_t size) {
    if (off + size + 5 > f.size()) throw std::runtime_error("block out of range");
    if (f[off + size] != 0)
        throw std::runtime_error("compressed index block not supported");
    return f.substr(off, size);
}

template <class F>
void for_each_entry(const std::string& block, F cb) {
    const uint8_t* d = (const uint8_t*)block.data();
    size_t n = block.size();
    uint32_t num_restarts;
    std::memcpy(&num_restarts, d + n - 4, 4);
    size_t limit = n - 4 - 4 * (size_t)num_restarts;
    Reader r{d, d + limit};
    std::string key;
    while (r.p < r.end) {
        uint64_t shared = r.varint(), non_shared = r.varint(), vlen = r.varint();
        key.resize(shared);
        key.append((const char*)r.p, non_shared); r.p += non_shared;
        std::string value((const char*)r.p, vlen); r.p += vlen;
        cb(key, value);
    }
}

std::string to_buffer_name(std::string name) {
    if (name.rfind("model/", 0) == 0) name = name.substr(6);
    std::string out;
    for (char c : name) {
        if (c == '/') out += "__";
        else out += c;
    }
    return out;
}

} // namespace

namespace gpt2 {

void convert_checkpoint(const std::string& model_dir, const std::string& out_path) {
    fs::path dir = model_dir;
    std::string idx = read_file(dir / "model.ckpt.index");
    if (idx.size() < 48) throw std::runtime_error("index file too small");

    // Footer: metaindex handle, index handle, padding, magic (48 bytes)
    Reader fr{(const uint8_t*)idx.data() + idx.size() - 48,
              (const uint8_t*)idx.data() + idx.size()};
    fr.varint(); fr.varint();
    uint64_t idx_off = fr.varint(), idx_size = fr.varint();

    std::map<std::string, Entry> entries;
    int num_shards = 1;

    for_each_entry(read_block(idx, idx_off, idx_size),
        [&](const std::string&, const std::string& handle) {
            Reader hr = make_reader(handle);
            uint64_t off = hr.varint(), size = hr.varint();
            for_each_entry(read_block(idx, off, size),
                [&](const std::string& k, const std::string& v) {
                    if (k.empty()) {                         // BundleHeaderProto
                        Reader r = make_reader(v);
                        while (r.p < r.end) {
                            uint64_t tag = r.varint();
                            if ((tag >> 3) == 1 && (tag & 7) == 0) num_shards = (int)r.varint();
                            else skip_field(r, tag & 7);
                        }
                    } else {
                        entries[k] = parse_entry(v);
                    }
                });
        });

    std::vector<std::ifstream> shards;
    for (int i = 0; i < num_shards; ++i) {
        char name[64];
        std::snprintf(name, sizeof(name), "model.ckpt.data-%05d-of-%05d", i, num_shards);
        shards.emplace_back(dir / name, std::ios::binary);
        if (!shards.back()) throw std::runtime_error(std::string("Cannot open ") + name);
    }

    torch::jit::Module module("GPT2Weights");
    for (auto& [name, e] : entries) {
        if (e.dtype != 1) continue;                          // float32 only (skips global_step)
        torch::Tensor t = torch::empty(e.shape, torch::kFloat32);
        if ((uint64_t)t.nbytes() != e.size)
            throw std::runtime_error("size mismatch for " + name);
        auto& f = shards.at(e.shard);
        f.seekg((std::streamoff)e.offset);
        f.read((char*)t.data_ptr(), (std::streamsize)e.size);
        if (!f) throw std::runtime_error("read failed for " + name);

        module.register_buffer(to_buffer_name(name), t.squeeze());   // squeeze == np.squeeze
    }
    module.save(out_path);
}

std::unordered_map<std::string, torch::Tensor> load_params(const std::string& path) {
    auto module = torch::jit::load(path, torch::kCPU);
    std::unordered_map<std::string, torch::Tensor> params;
    for (const auto& b : module.named_buffers())
        params[b.name] = b.value;
    return params;
}

} // namespace gpt2

// DATASETS should end with '/', e.g. "/home/moinshaikh/CLionProjects/LargeLanguageModelcpp/"
static const std::string MODEL_DIR = std::string(DATASETS_DIR) + "gpt2/124M";
static const std::string OUT_PATH  = std::string(DATASETS_DIR) + "gpt2/gpt2_124M.torchscript";

TEST_CASE("convert GPT-2 124M checkpoint to torchscript") {
    REQUIRE_NOTHROW(gpt2::convert_checkpoint(MODEL_DIR, OUT_PATH));
    CHECK(std::filesystem::exists(OUT_PATH));
}

TEST_CASE("converted params have expected shapes") {
    auto p = gpt2::load_params(OUT_PATH);

    CHECK(p.at("wte").sizes()  == torch::IntArrayRef({50257, 768}));
    CHECK(p.at("wpe").sizes()  == torch::IntArrayRef({1024, 768}));
    CHECK(p.at("ln_f__g").sizes() == torch::IntArrayRef({768}));

    CHECK(p.at("h0__attn__c_attn__w").sizes()  == torch::IntArrayRef({768, 2304}));
    CHECK(p.at("h0__attn__c_attn__b").sizes()  == torch::IntArrayRef({2304}));
    CHECK(p.at("h11__mlp__c_fc__w").sizes()    == torch::IntArrayRef({768, 3072}));
    CHECK(p.at("h11__mlp__c_proj__w").sizes()  == torch::IntArrayRef({3072, 768}));
    CHECK(p.at("h5__ln_1__g").sizes()          == torch::IntArrayRef({768}));
}
