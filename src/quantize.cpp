#include "calibrator/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace calibrator {

int qmax_for_bits(int bits) {
    return (1 << (bits - 1)) - 1;
}

bool is_supported_bits(int bits) {
    return bits == 2 || bits == 4 || bits == 8;
}

QuantizedMatrix quantize_grouped(MatrixRef values, int bits, int group_size) {
    if (!is_supported_bits(bits)) {
        throw std::invalid_argument("bits must be one of {2, 4, 8}");
    }
    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    if (!values.allFinite()) {
        throw std::invalid_argument("values contain non-finite entries");
    }

    const int rows = values.rows();
    const int cols = values.cols();
    const int num_groups = (cols + group_size - 1) / group_size;
    const int qmax = qmax_for_bits(bits);

    QuantizedMatrix out;
    out.rows = rows;
    out.cols = cols;
    out.bits = bits;
    out.group_size = group_size;
    out.codes.assign(static_cast<size_t>(rows) * cols, 0);
    out.scales.assign(static_cast<size_t>(rows) * num_groups, 0.0f);

    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < num_groups; ++g) {
            const int begin = g * group_size;
            const int end = std::min(begin + group_size, cols);
            float amax = 0.0f;
            for (int c = begin; c < end; ++c) {
                amax = std::max(amax, std::fabs(values(r, c)));
            }
            const size_t scale_index = static_cast<size_t>(r) * num_groups + g;
            if (amax == 0.0f) {
                // All-zero group: zero scale, zero codes -> dequantizes to zero.
                out.scales[scale_index] = 0.0f;
                continue;
            }
            const float scale = amax / static_cast<float>(qmax);
            out.scales[scale_index] = scale;
            for (int c = begin; c < end; ++c) {
                float q = std::nearbyint(values(r, c) / scale);
                q = std::min(static_cast<float>(qmax), std::max(static_cast<float>(-qmax), q));
                out.codes[static_cast<size_t>(r) * cols + c] = static_cast<int32_t>(q);
            }
        }
    }
    return out;
}

Matrix dequantize_grouped(const QuantizedMatrix& quantized) {
    if (!is_supported_bits(quantized.bits)) {
        throw std::invalid_argument("bits must be one of {2, 4, 8}");
    }
    Matrix out = Matrix::Zero(quantized.rows, quantized.cols);
    const int num_groups = quantized.num_groups();
    for (int r = 0; r < quantized.rows; ++r) {
        for (int g = 0; g < num_groups; ++g) {
            const int begin = g * quantized.group_size;
            const int end = std::min(begin + quantized.group_size, quantized.cols);
            const float scale = quantized.scales[static_cast<size_t>(r) * num_groups + g];
            for (int c = begin; c < end; ++c) {
                out(r, c) = static_cast<float>(
                    quantized.codes[static_cast<size_t>(r) * quantized.cols + c]) * scale;
            }
        }
    }
    return out;
}

ErrorStats quantization_error(MatrixRef reference, MatrixRef dequantized) {
    if (reference.rows() != dequantized.rows() || reference.cols() != dequantized.cols()) {
        throw std::invalid_argument("error stats require matching shapes");
    }
    ErrorStats stats;
    const double n = static_cast<double>(reference.size());
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double ref_sq = 0.0;
    for (int j = 0; j < reference.cols(); ++j) {
        for (int i = 0; i < reference.rows(); ++i) {
            const double d = static_cast<double>(dequantized(i, j)) -
                             static_cast<double>(reference(i, j));
            const double ad = std::fabs(d);
            sum_sq += d * d;
            sum_abs += ad;
            stats.max_abs = std::max(stats.max_abs, ad);
            const double rv = static_cast<double>(reference(i, j));
            ref_sq += rv * rv;
        }
    }
    stats.rmse = std::sqrt(sum_sq / n);
    stats.mean_abs = sum_abs / n;
    stats.relative_rmse = ref_sq > 0.0 ? std::sqrt(sum_sq / ref_sq) : 0.0;
    return stats;
}

Matrix simulate_quantized_linear(MatrixRef rotated_activations,
                                 MatrixRef transformed_weight, int activation_bits,
                                 int weight_bits, int activation_group_size,
                                 int weight_group_size) {
    if (rotated_activations.cols() != transformed_weight.rows()) {
        throw std::invalid_argument(
            "rotated activation columns must match transformed weight rows");
    }
    const QuantizedMatrix qx =
        quantize_grouped(rotated_activations, activation_bits, activation_group_size);
    const QuantizedMatrix qw =
        quantize_grouped(transformed_weight, weight_bits, weight_group_size);
    const Matrix dx = dequantize_grouped(qx);
    const Matrix dw = dequantize_grouped(qw);
    return dx * dw;
}

}  // namespace calibrator
