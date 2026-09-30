#include "oqc/calibrator.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <fstream>
#include <stdexcept>

namespace oqc {

namespace {

constexpr char kCalibMagic[8] = {'O', 'Q', 'C', 'C', 'A', 'L', 'I', 'B'};
constexpr char kQuantMagic[8] = {'O', 'Q', 'C', 'Q', 'U', 'A', 'N', 'T'};
constexpr uint32_t kVersion = 1;

uint32_t fnv1a32(const uint8_t* data, std::size_t n) {
    uint32_t h = 2166136261u;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

void put_u16le(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
}

void put_u32le(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void put_u64le(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        b.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void put_f64le(std::vector<uint8_t>& b, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, 8);
    put_u64le(b, bits);
}

uint16_t rd_u16le(const std::vector<uint8_t>& b, std::size_t& p) {
    if (p + 2 > b.size()) throw CalibError("archive truncated");
    uint16_t v = static_cast<uint16_t>(b[p]) |
                 (static_cast<uint16_t>(b[p + 1]) << 8);
    p += 2;
    return v;
}

uint32_t rd_u32le(const std::vector<uint8_t>& b, std::size_t& p) {
    if (p + 4 > b.size()) throw CalibError("archive truncated");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<uint32_t>(b[p + i]) << (8 * i);
    }
    p += 4;
    return v;
}

uint64_t rd_u64le(const std::vector<uint8_t>& b, std::size_t& p) {
    if (p + 8 > b.size()) throw CalibError("archive truncated");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(b[p + i]) << (8 * i);
    }
    p += 8;
    return v;
}

double rd_f64le(const std::vector<uint8_t>& b, std::size_t& p) {
    uint64_t bits = rd_u64le(b, p);
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

void require(bool cond, const char* msg) {
    if (!cond) throw CalibError(msg);
}

void put_matrix(std::vector<uint8_t>& b, const Matrix& m) {
    put_u16le(b, static_cast<uint16_t>(m.rows()));
    put_u16le(b, static_cast<uint16_t>(m.cols()));
    for (int j = 0; j < m.cols(); ++j) {
        for (int i = 0; i < m.rows(); ++i) {
            put_f64le(b, m(i, j));
        }
    }
}

Matrix rd_matrix(const std::vector<uint8_t>& b, std::size_t& p) {
    int rows = rd_u16le(b, p);
    int cols = rd_u16le(b, p);
    require(rows >= 1 && rows <= kMaxDim && cols >= 1 && cols <= kMaxDim,
            "archive matrix dims out of range");
    Matrix m(rows, cols);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            m(i, j) = rd_f64le(b, p);
        }
    }
    require(m.allFinite(), "archive matrix contains non-finite values");
    return m;
}

void write_file(const std::string& path, std::vector<uint8_t> data) {
    put_u32le(data, fnv1a32(data.data(), data.size()));
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw CalibError("cannot open file for write: " + path);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    if (!f) throw CalibError("write failed: " + path);
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw CalibError("cannot open file for read: " + path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    require(data.size() >= 12, "file too small to be an archive");
    uint32_t stored;
    std::memcpy(&stored, data.data() + data.size() - 4, 4);
    uint32_t actual = fnv1a32(data.data(), data.size() - 4);
    require(stored == actual, "archive checksum mismatch");
    data.resize(data.size() - 4);
    return data;
}

void check_magic(const std::vector<uint8_t>& b, const char* magic) {
    require(b.size() >= 8, "archive too small");
    require(std::memcmp(b.data(), magic, 8) == 0, "bad archive magic");
}

}  // namespace

void save_calibration(const std::string& path,
                      const CalibResult& result,
                      const Matrix& activations,
                      const Matrix& weights) {
    validate_finite(activations, "activations");
    validate_finite(weights, "weights");
    validate_rotation(result.rotation);
    require(std::isfinite(result.loss), "cannot save non-finite loss");
    require(activations.rows() == result.rotation.rows(),
            "saved activation dims conflict with rotation");
    require(weights.cols() == result.rotation.rows(),
            "saved weight cols conflict with rotation");
    require(!result.history.empty(), "history must contain valid entries");

    std::vector<uint8_t> b;
    b.insert(b.end(), kCalibMagic, kCalibMagic + 8);
    put_u32le(b, kVersion);
    put_matrix(b, activations);
    put_matrix(b, weights);
    put_matrix(b, result.rotation);
    put_f64le(b, result.loss);
    put_u32le(b, static_cast<uint32_t>(result.accepted_iters));
    put_u32le(b, static_cast<uint32_t>(result.history.size()));
    for (const auto& e : result.history) {
        put_u32le(b, static_cast<uint32_t>(e.iter));
        put_f64le(b, e.loss);
        put_matrix(b, e.rotation);
    }
    write_file(path, b);
}

CalibArchive load_calibration(const std::string& path) {
    std::vector<uint8_t> b = read_file(path);
    check_magic(b, kCalibMagic);
    std::size_t p = 8;
    uint32_t version = rd_u32le(b, p);
    require(version == kVersion, "unsupported calibration archive version");

    CalibArchive ar;
    ar.activations = rd_matrix(b, p);
    ar.weights = rd_matrix(b, p);
    ar.result.rotation = rd_matrix(b, p);
    ar.result.loss = rd_f64le(b, p);
    require(std::isfinite(ar.result.loss), "loaded loss non-finite");
    ar.result.accepted_iters = static_cast<int>(rd_u32le(b, p));
    uint32_t nh = rd_u32le(b, p);
    require(nh > 0, "loaded history empty");
    require(nh <= static_cast<uint32_t>(kMaxDim) * 100000u + 1u,
            "history length implausible");
    ar.result.history.reserve(nh);
    double best = std::numeric_limits<double>::infinity();
    for (uint32_t i = 0; i < nh; ++i) {
        HistoryEntry e;
        e.iter = static_cast<int>(rd_u32le(b, p));
        e.loss = rd_f64le(b, p);
        e.rotation = rd_matrix(b, p);
        require(std::isfinite(e.loss), "history loss non-finite");
        ar.result.history.push_back(e);
        best = std::min(best, e.loss);
    }
    require(p == b.size(), "archive has trailing bytes");
    require(ar.activations.rows() == ar.result.rotation.rows(),
            "loaded activation dims conflict");
    require(ar.weights.cols() == ar.result.rotation.rows(),
            "loaded weight dims conflict");
    validate_rotation(ar.result.rotation);
    double tol = 1e-12 * std::max(1.0, std::fabs(best));
    require(ar.result.loss <= best + tol,
            "loaded result loss is worse than the best history entry");
    double min_history = std::numeric_limits<double>::infinity();
    for (const auto& e : ar.result.history) {
        min_history = std::min(min_history, e.loss);
    }
    require(std::fabs(min_history - best) <= tol,
            "history best loss inconsistent");
    return ar;
}

void save_quantized(const std::string& path, const QuantTensor& q) {
    if (q.bits != 2 && q.bits != 4 && q.bits != 8) {
        throw CalibError("bits must be one of {2,4,8}");
    }
    if (q.rows <= 0 || q.rows > kMaxDim || q.cols <= 0 || q.cols > kMaxDim) {
        throw CalibError("quant tensor shape out of range");
    }
    if (q.codes.size() !=
        static_cast<std::size_t>(q.rows) * q.cols) {
        throw CalibError("quant tensor code count mismatch");
    }
    if (q.group_size <= 0 || q.group_size > kMaxDim ||
        (q.axis != 0 && q.axis != 1)) {
        throw CalibError("quant tensor grouping invalid");
    }
    int grouped_dim = q.axis == 0 ? q.rows : q.cols;
    int outer = q.axis == 0 ? q.cols : q.rows;
    int ng = (grouped_dim + q.group_size - 1) / q.group_size;
    if (q.scales.size() != static_cast<std::size_t>(outer) * ng) {
        throw CalibError("quant tensor scale count mismatch");
    }
    int levels = (1 << (q.bits - 1)) - 1;
    for (std::int32_t c : q.codes) {
        if (c < -levels || c > levels) {
            throw CalibError("quant code outside signed range");
        }
    }

    std::vector<uint8_t> b;
    b.insert(b.end(), kQuantMagic, kQuantMagic + 8);
    put_u32le(b, kVersion);
    put_u16le(b, static_cast<uint16_t>(q.bits));
    put_u16le(b, static_cast<uint16_t>(q.group_size));
    put_u16le(b, static_cast<uint16_t>(q.axis));
    put_u16le(b, static_cast<uint16_t>(q.rows));
    put_u16le(b, static_cast<uint16_t>(q.cols));
    put_u32le(b, static_cast<uint32_t>(q.codes.size()));
    for (std::int32_t c : q.codes) {
        put_u32le(b, static_cast<uint32_t>(c));
    }
    put_u32le(b, static_cast<uint32_t>(q.scales.size()));
    for (double s : q.scales) {
        require(std::isfinite(s) && s >= 0.0, "scale must be finite >= 0");
        put_f64le(b, s);
    }
    write_file(path, b);
}

QuantTensor load_quantized(const std::string& path) {
    std::vector<uint8_t> b = read_file(path);
    check_magic(b, kQuantMagic);
    std::size_t p = 8;
    uint32_t version = rd_u32le(b, p);
    require(version == kVersion, "unsupported quant archive version");
    QuantTensor q;
    q.bits = rd_u16le(b, p);
    q.group_size = rd_u16le(b, p);
    q.axis = rd_u16le(b, p);
    q.rows = rd_u16le(b, p);
    q.cols = rd_u16le(b, p);
    require(q.bits == 2 || q.bits == 4 || q.bits == 8, "bits not in {2,4,8}");
    require(q.group_size >= 1 && q.group_size <= kMaxDim, "bad group size");
    require(q.axis == 0 || q.axis == 1, "bad quant axis");
    require(q.rows >= 1 && q.rows <= kMaxDim, "rows out of range");
    require(q.cols >= 1 && q.cols <= kMaxDim, "cols out of range");
    int grouped_dim = q.axis == 0 ? q.rows : q.cols;
    int outer = q.axis == 0 ? q.cols : q.rows;
    require(q.group_size <= grouped_dim, "group size exceeds dim");
    int ng = (grouped_dim + q.group_size - 1) / q.group_size;

    uint32_t ncode = rd_u32le(b, p);
    require(ncode == static_cast<uint32_t>(q.rows) * q.cols,
            "code count mismatch");
    int levels = (1 << (q.bits - 1)) - 1;
    q.codes.resize(ncode);
    for (uint32_t i = 0; i < ncode; ++i) {
        int32_t c = static_cast<int32_t>(rd_u32le(b, p));
        require(c >= -levels && c <= levels, "code outside signed range");
        q.codes[i] = c;
    }
    uint32_t nscale = rd_u32le(b, p);
    require(nscale == static_cast<uint32_t>(outer) * ng, "scale count mismatch");
    q.scales.resize(nscale);
    for (uint32_t i = 0; i < nscale; ++i) {
        double s = rd_f64le(b, p);
        require(std::isfinite(s) && s >= 0.0, "bad scale value");
        q.scales[i] = s;
    }
    require(p == b.size(), "quant archive has trailing bytes");
    return q;
}

}  // namespace oqc
