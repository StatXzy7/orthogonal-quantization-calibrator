#include "oqc/calibrator.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace oqc {

namespace {

int levels_for(int bits) {
    return (1 << (bits - 1)) - 1;
}

std::int32_t clamp_round(double x, int levels) {
    long q = std::lround(x);
    if (q > levels) {
        q = levels;
    }
    if (q < -levels) {
        q = -levels;
    }
    return static_cast<std::int32_t>(q);
}

}  // namespace

QuantTensor quantize_symmetric(const Matrix& m, const QuantConfig& config) {
    validate_finite(m, "matrix");
    const int grouped_dim = config.axis == 0 ? m.rows() : m.cols();
    validate_quant_config(config, grouped_dim);

    const int levels = levels_for(config.bits);
    const int outer = config.axis == 0 ? m.cols() : m.rows();
    const int ng = (grouped_dim + config.group_size - 1) / config.group_size;

    QuantTensor q;
    q.bits = config.bits;
    q.group_size = config.group_size;
    q.axis = config.axis;
    q.rows = m.rows();
    q.cols = m.cols();
    q.codes.assign(static_cast<std::size_t>(m.rows()) * m.cols(), 0);
    q.scales.assign(static_cast<std::size_t>(outer) * ng, 0.0);

    for (int line = 0; line < outer; ++line) {
        for (int g = 0; g < ng; ++g) {
            int begin = g * config.group_size;
            int end = std::min(grouped_dim, begin + config.group_size);

            double amax = 0.0;
            for (int t = begin; t < end; ++t) {
                double v = config.axis == 0 ? m(t, line) : m(line, t);
                amax = std::max(amax, std::fabs(v));
            }
            double scale = amax == 0.0 ? 0.0 : amax / levels;
            q.scales[static_cast<std::size_t>(line) * ng + g] = scale;

            for (int t = begin; t < end; ++t) {
                int row = config.axis == 0 ? t : line;
                int col = config.axis == 0 ? line : t;
                double v = m(row, col);
                std::int32_t code =
                    scale == 0.0 ? 0 : clamp_round(v / scale, levels);
                q.row_begin(row)[col] = code;
            }
        }
    }
    return q;
}

Matrix dequantize(const QuantTensor& q) {
    if (q.rows <= 0 || q.cols <= 0 || q.rows > kMaxDim || q.cols > kMaxDim) {
        throw CalibError("quant tensor has invalid shape");
    }
    if (q.codes.size() !=
        static_cast<std::size_t>(q.rows) * q.cols) {
        throw CalibError("quant tensor code count mismatch");
    }
    if (q.bits != 2 && q.bits != 4 && q.bits != 8) {
        throw CalibError("quant tensor bits must be in {2,4,8}");
    }
    if (q.group_size <= 0 || q.group_size > kMaxDim ||
        (q.axis != 0 && q.axis != 1)) {
        throw CalibError("quant tensor grouping invalid");
    }
    const int grouped_dim = q.axis == 0 ? q.rows : q.cols;
    const int outer = q.axis == 0 ? q.cols : q.rows;
    const int ng = (grouped_dim + q.group_size - 1) / q.group_size;
    if (q.scales.size() != static_cast<std::size_t>(outer) * ng) {
        throw CalibError("quant tensor scale count mismatch");
    }
    const int levels = levels_for(q.bits);
    for (std::int32_t c : q.codes) {
        if (c < -levels || c > levels) {
            throw CalibError("quant code outside signed symmetric range");
        }
    }

    Matrix dq(q.rows, q.cols);
    for (int r = 0; r < q.rows; ++r) {
        for (int c = 0; c < q.cols; ++c) {
            int line = q.axis == 0 ? c : r;
            int t = q.axis == 0 ? r : c;
            int g = t / q.group_size;
            double scale = q.scales[static_cast<std::size_t>(line) * ng + g];
            dq(r, c) = static_cast<double>(q.row_begin(r)[c]) * scale;
        }
    }
    return dq;
}

QuantStats quantize_with_stats(const Matrix& m, const QuantConfig& config) {
    QuantTensor q = quantize_symmetric(m, config);
    QuantStats stats;
    stats.dequantized = dequantize(q);
    Matrix diff = stats.dequantized - m;
    stats.max_abs_error = diff.cwiseAbs().maxCoeff();
    stats.mean_abs_error = diff.cwiseAbs().mean();
    stats.rmse = std::sqrt(diff.array().square().mean());
    return stats;
}

}  // namespace oqc
