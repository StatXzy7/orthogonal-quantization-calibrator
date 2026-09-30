#include "test_util.hpp"

using namespace calibrator;

TEST_CASE("quantize/integer_ranges_for_all_bit_widths") {
    CHECK(qmax_for_bits(2) == 1);
    CHECK(qmax_for_bits(4) == 7);
    CHECK(qmax_for_bits(8) == 127);
    Matrix m(1, 5);
    m << -10.0f, -5.0f, 0.0f, 5.0f, 10.0f;
    for (int bits : {2, 4, 8}) {
        QuantizedMatrix q = quantize_grouped(m, bits, 5);
        const int qmax = qmax_for_bits(bits);
        for (int32_t code : q.codes) {
            CHECK(code >= -qmax);
            CHECK(code <= qmax);
        }
        CHECK(std::fabs(q.scales[0] - 10.0f / qmax) < 1e-6);
        // Extremes map exactly to +/- qmax.
        CHECK(q.codes[0] == -qmax);
        CHECK(q.codes[4] == qmax);
        Matrix dq = dequantize_grouped(q);
        ErrorStats stats = quantization_error(m, dq);
        CHECK(stats.max_abs <= q.scales[0] * 0.5f + 1e-6f);
    }
}

TEST_CASE("quantize/all_zero_group_restores_zero") {
    Matrix m = Matrix::Zero(3, 7);
    QuantizedMatrix q = quantize_grouped(m, 4, 3);
    for (float scale : q.scales) {
        CHECK(scale == 0.0f);
    }
    for (int32_t code : q.codes) {
        CHECK(code == 0);
    }
    Matrix dq = dequantize_grouped(q);
    CHECK_CLOSE(max_abs_diff(m, dq), 0.0, 0.0);
    ErrorStats stats = quantization_error(m, dq);
    CHECK(stats.rmse == 0.0);
    CHECK(stats.max_abs == 0.0);
    CHECK(stats.relative_rmse == 0.0);
}

TEST_CASE("quantize/trailing_short_group_is_exact") {
    // 5 columns, group size 3 -> groups of 3 and 2.
    Matrix m(2, 5);
    m << 3.0f, -6.0f, 0.0f, 1.5f, -1.5f,
         0.0f, 0.0f, 0.0f, 9.0f, -3.0f;
    QuantizedMatrix q = quantize_grouped(m, 8, 3);
    CHECK(q.num_groups() == 2);
    // Trailing group of row 0: amax = 1.5, both values exactly representable.
    const float trailing_scale = 1.5f / 127.0f;
    CHECK(std::fabs(q.scales[0 * 2 + 1] - trailing_scale) < 1e-6f);
    Matrix dq = dequantize_grouped(q);
    CHECK(dq.allFinite());
    CHECK_CLOSE(std::fabs(dq(0, 3) - 1.5f), 0.0, 1e-6);
    CHECK_CLOSE(std::fabs(dq(0, 4) + 1.5f), 0.0, 1e-6);
    CHECK_CLOSE(std::fabs(dq(1, 3) - 9.0f), 0.0, 1e-6);
    // row 1 trailing group shares scale 9/127, so -3 lands within half a scale (~0.036)
    CHECK(std::fabs(dq(1, 4) + 3.0f) < 0.04f);
    ErrorStats stats = quantization_error(m, dq);
    CHECK(stats.max_abs < 0.03f);
}

TEST_CASE("quantize/group_size_one_is_per_channel_scale") {
    Matrix m(1, 3);
    m << 100.0f, 1.0f, 0.1f;
    QuantizedMatrix q = quantize_grouped(m, 8, 1);
    CHECK(q.num_groups() == 3);
    Matrix dq = dequantize_grouped(q);
    ErrorStats stats = quantization_error(m, dq);
    CHECK(stats.relative_rmse < 0.01);
}

TEST_CASE("quantize/higher_bit_width_is_no_worse") {
    std::mt19937_64 rng(11);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix m(64, 16);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 16; ++j) {
            m(i, j) = normal(rng) * (j + 1);
        }
    }
    double prev = std::numeric_limits<double>::infinity();
    for (int bits : {2, 4, 8}) {
        QuantizedMatrix q = quantize_grouped(m, bits, 4);
        Matrix dq = dequantize_grouped(q);
        double err = quantization_error(m, dq).rmse;
        CHECK(err <= prev + 1e-6);
        prev = err;
    }
}
