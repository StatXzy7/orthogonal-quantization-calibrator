#pragma once

#include <cstdint>
#include <vector>

#include "calibrator/quantize.hpp"

namespace calibrator {

// Bit-level packing of signed symmetric integer codes.
//
// Codes are stored in `bits`-wide little-endian bit fields (bit 0 of byte 0 is
// the LSB of the first code). Fields never straddle padding: each row is padded
// to a whole byte, so short trailing groups / rows round-trip exactly.
std::vector<uint8_t> pack_codes(const QuantizedMatrix& quantized);

// Rebuild codes from a packed buffer. The exact number of expected codes
// (rows * cols) must be known from the accompanying metadata.
std::vector<int32_t> unpack_codes(const std::vector<uint8_t>& packed, int bits,
                                  int rows, int cols);

}  // namespace calibrator
