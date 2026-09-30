#include "calibrator/io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace calibrator {
namespace {

constexpr char kMagic[4] = {'O', 'Q', 'C', 'B'};
constexpr uint16_t kVersion = 1;
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t hash = kFnvOffset;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

class Writer {
public:
    void u8(uint8_t v) { buf_.push_back(v); }
    void u16(uint16_t v) {
        buf_.push_back(static_cast<uint8_t>(v & 0xff));
        buf_.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
        }
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf_.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
        }
    }
    void f32(float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        u32(bits);
    }
    void f64(double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        u64(bits);
    }
    const std::vector<uint8_t>& bytes() const { return buf_; }

private:
    std::vector<uint8_t> buf_;
};

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& data) : data_(data) {}

    uint8_t u8() {
        need(1);
        return data_[pos_++];
    }
    uint16_t u16() {
        need(2);
        uint16_t v = 0;
        for (int i = 0; i < 2; ++i) {
            v |= static_cast<uint16_t>(data_[pos_ + i]) << (8 * i);
        }
        pos_ += 2;
        return v;
    }
    uint32_t u32() {
        need(4);
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(data_[pos_ + i]) << (8 * i);
        }
        pos_ += 4;
        return v;
    }
    uint64_t u64() {
        need(8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
        }
        pos_ += 8;
        return v;
    }
    float f32() {
        const uint32_t bits = u32();
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    double f64() {
        const uint64_t bits = u64();
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }

private:
    void need(size_t n) const {
        if (pos_ + n > data_.size()) {
            throw std::runtime_error("calibration file is truncated");
        }
    }
    const std::vector<uint8_t>& data_;
    size_t pos_ = 0;
};

}  // namespace

void save_calibration(const std::string& path, const CalibrationResult& result,
                      const CalibrationConfig& config) {
    if (result.rotation.rows() != result.rotation.cols() ||
        !result.rotation.allFinite()) {
        throw std::invalid_argument("refusing to save a non-finite / non-square rotation");
    }
    const int n = result.rotation.rows();
    if (n <= 0 || n > kMaxDim) {
        throw std::invalid_argument("rotation dimension out of range when saving");
    }

    Writer w;
    w.u8(static_cast<uint8_t>(kMagic[0]));
    w.u8(static_cast<uint8_t>(kMagic[1]));
    w.u8(static_cast<uint8_t>(kMagic[2]));
    w.u8(static_cast<uint8_t>(kMagic[3]));
    w.u16(kVersion);
    w.u32(static_cast<uint32_t>(n));
    w.u32(static_cast<uint32_t>(config.max_iterations));
    w.u64(config.seed);
    w.f64(result.initial_loss);
    w.f64(result.final_loss);
    w.u32(static_cast<uint32_t>(result.iterations_run));
    w.u32(static_cast<uint32_t>(result.accepted_steps));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            w.f32(result.rotation(i, j));
        }
    }
    const uint64_t checksum = fnv1a(w.bytes().data(), w.bytes().size());
    w.u64(checksum);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open calibration file for writing: " + path);
    }
    out.write(reinterpret_cast<const char*>(w.bytes().data()),
              static_cast<std::streamsize>(w.bytes().size()));
    if (!out) {
        throw std::runtime_error("failed to write calibration file: " + path);
    }
}

CalibrationArtifact load_calibration(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open calibration file for reading: " + path);
    }
    const std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to determine calibration file size: " + path);
    }
    in.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("failed to read calibration file: " + path);
    }

    constexpr size_t kPreamble = 4 + 2;
    constexpr size_t kFixedHeader = 4 + 4 + 8 + 8 + 8 + 4 + 4;
    if (data.size() < kPreamble + kFixedHeader + 8) {
        throw std::runtime_error("calibration file is truncated");
    }
    if (std::memcmp(data.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not an OQCB calibration file (bad magic)");
    }
    Reader reader(data);
    reader.u8();
    reader.u8();
    reader.u8();
    reader.u8();
    const uint16_t version = reader.u16();
    if (version != kVersion) {
        throw std::runtime_error("unsupported calibration file version");
    }
    const uint32_t n = reader.u32();
    if (n == 0 || n > static_cast<uint32_t>(kMaxDim)) {
        throw std::runtime_error("calibration file has an invalid rotation dimension");
    }
    const size_t expected_size =
        kPreamble + kFixedHeader + static_cast<size_t>(n) * n * 4 + 8;
    if (data.size() != expected_size) {
        throw std::runtime_error("calibration file length does not match its header");
    }

    CalibrationArtifact artifact;
    artifact.config.max_iterations = static_cast<int>(reader.u32());
    artifact.config.seed = reader.u64();
    artifact.result.initial_loss = reader.f64();
    artifact.result.final_loss = reader.f64();
    artifact.result.iterations_run = static_cast<int>(reader.u32());
    artifact.result.accepted_steps = static_cast<int>(reader.u32());
    artifact.result.rotation = Matrix(static_cast<int>(n), static_cast<int>(n));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            artifact.result.rotation(static_cast<int>(i), static_cast<int>(j)) = reader.f32();
        }
    }
    const uint64_t expected_checksum = fnv1a(data.data(), data.size() - 8);
    const uint64_t actual_checksum = reader.u64();
    if (expected_checksum != actual_checksum) {
        throw std::runtime_error("calibration file checksum mismatch (corrupted)");
    }
    if (!artifact.result.rotation.allFinite()) {
        throw std::runtime_error("calibration file contains non-finite rotation entries");
    }
    if (!is_orthogonal(artifact.result.rotation)) {
        throw std::runtime_error("calibration file rotation is not orthogonal");
    }
    if (artifact.config.max_iterations < 0 ||
        !std::isfinite(artifact.result.initial_loss) ||
        !std::isfinite(artifact.result.final_loss)) {
        throw std::runtime_error("calibration file contains invalid scalar fields");
    }
    return artifact;
}

}  // namespace calibrator
