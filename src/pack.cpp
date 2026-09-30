#include "calibrator/pack.hpp"

#include <stdexcept>

namespace calibrator {
namespace {

uint32_t low_mask(int bits) {
    return (1u << bits) - 1u;
}

}  // namespace

std::vector<uint8_t> pack_codes(const QuantizedMatrix& quantized) {
    const int bits = quantized.bits;
    if (!is_supported_bits(bits)) {
        throw std::invalid_argument("bits must be one of {2, 4, 8}");
    }
    const int rows = quantized.rows;
    const int cols = quantized.cols;
    const size_t expected = static_cast<size_t>(rows) * cols;
    if (quantized.codes.size() != expected) {
        throw std::invalid_argument("codes buffer does not match rows*cols");
    }
    const int qmax = qmax_for_bits(bits);
    const uint32_t mask = low_mask(bits);
    const size_t bytes_per_row = (static_cast<size_t>(cols) * bits + 7u) / 8u;

    std::vector<uint8_t> packed(rows * bytes_per_row, 0u);
    for (int r = 0; r < rows; ++r) {
        uint8_t* row_bytes = packed.data() + static_cast<size_t>(r) * bytes_per_row;
        for (int c = 0; c < cols; ++c) {
            const int32_t code = quantized.codes[static_cast<size_t>(r) * cols + c];
            if (code < -qmax || code > qmax) {
                throw std::invalid_argument("code out of range for the declared bit width");
            }
            const auto field = static_cast<uint32_t>(code) & mask;
            const size_t bit_pos = static_cast<size_t>(c) * bits;
            for (int b = 0; b < bits; ++b) {
                if (field & (1u << b)) {
                    row_bytes[(bit_pos + b) / 8u] |=
                        static_cast<uint8_t>(1u << ((bit_pos + b) % 8u));
                }
            }
        }
    }
    return packed;
}

std::vector<int32_t> unpack_codes(const std::vector<uint8_t>& packed, int bits,
                                  int rows, int cols) {
    if (!is_supported_bits(bits)) {
        throw std::invalid_argument("bits must be one of {2, 4, 8}");
    }
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("rows and cols must be non-negative");
    }
    const size_t bytes_per_row = (static_cast<size_t>(cols) * bits + 7u) / 8u;
    const size_t needed = static_cast<size_t>(rows) * bytes_per_row;
    if (packed.size() < needed) {
        throw std::invalid_argument("packed buffer is too short for the declared shape");
    }

    const uint32_t mask = low_mask(bits);
    const uint32_t sign_bit = 1u << (bits - 1);
    std::vector<int32_t> codes(static_cast<size_t>(rows) * cols, 0);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_bytes = packed.data() + static_cast<size_t>(r) * bytes_per_row;
        for (int c = 0; c < cols; ++c) {
            const size_t bit_pos = static_cast<size_t>(c) * bits;
            uint32_t field = 0;
            for (int b = 0; b < bits; ++b) {
                const size_t pos = bit_pos + b;
                if (row_bytes[pos / 8u] & (1u << (pos % 8u))) {
                    field |= 1u << b;
                }
            }
            int32_t code = static_cast<int32_t>(field);
            if (field & sign_bit) {
                code = static_cast<int32_t>(field | ~mask);
            }
            codes[static_cast<size_t>(r) * cols + c] = code;
        }
    }
    return codes;
}

}  // namespace calibrator
