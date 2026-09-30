#pragma once

#include <cstdint>
#include <string>

#include "calibrator/calibration.hpp"

namespace calibrator {

// Supported bit widths: 2, 4, 8. Integer level range is symmetric
// [-qmax, qmax] with qmax = 2^(b-1) - 1.
int qmax_for_bits(int bits);
bool is_supported_bits(int bits);

struct QuantizedMatrix {
    int rows = 0;
    int cols = 0;
    int bits = 2;
    int group_size = 1;            // columns per scale group
    std::vector<int32_t> codes;    // rows * cols, in [-qmax, qmax]
    std::vector<float> scales;     // rows * num_groups, one scale per group

    int num_groups() const { return cols == 0 ? 0 : (cols + group_size - 1) / group_size; }
};

struct ErrorStats {
    double rmse = 0.0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double relative_rmse = 0.0;  // rmse / rms(reference); 0 when reference is zero
};

// Signed symmetric per-row grouped quantization. Groups are contiguous blocks
// of `group_size` columns; the final group may be shorter. All-zero groups
// quantize to zero with scale 0 and dequantize back to all zeros.
QuantizedMatrix quantize_grouped(MatrixRef values, int bits, int group_size);

// Dequantize: value_hat = code * scale for every element.
Matrix dequantize_grouped(const QuantizedMatrix& quantized);

// Element-wise dequantization error statistics against a reference.
ErrorStats quantization_error(MatrixRef reference, MatrixRef dequantized);

// Quantize activations and transformed weight, then simulate the rotated
// linear layer: Xr_deq * Wt_deq.
Matrix simulate_quantized_linear(MatrixRef rotated_activations,
                                 MatrixRef transformed_weight, int activation_bits,
                                 int weight_bits, int activation_group_size,
                                 int weight_group_size);

}  // namespace calibrator
