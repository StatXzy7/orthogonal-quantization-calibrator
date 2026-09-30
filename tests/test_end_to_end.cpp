#include <cstdio>

#include "test_util.hpp"

using namespace calibrator;

TEST_CASE("e2e/full_calibrate_quantize_pack_io_pipeline") {
    const int n = 8;
    const int out = 6;
    const int rows = 256;
    std::mt19937_64 rng(2026);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Skewed activations with a couple of extreme channels.
    Matrix x(rows, n);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            const float scale = (j == 0) ? 25.0f : (j == 1) ? 0.01f : 0.3f;
            x(i, j) = normal(rng) * scale;
        }
    }
    Matrix w(n, out);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < out; ++j) {
            w(i, j) = normal(rng);
        }
    }

    CalibrationConfig config{128, 31337};
    Matrix r0 = Matrix::Identity(n, n);
    CalibrationResult result = calibrate(x, w, r0, config);
    CHECK(result.rotation.allFinite());
    CHECK(max_orthogonality_error(result.rotation) < 1e-5);
    CHECK(result.final_loss < result.initial_loss);

    // Unquantized rotated linear map equals the original.
    Matrix xr = rotate_activations(x, result.rotation);
    Matrix wt = transform_weight(w, result.rotation);
    CHECK(relative_fro_diff(x * w, xr * wt) < 1e-4);

    // Separate grouped quantization (group sizes chosen by the caller).
    const int a_bits = 4;
    const int w_bits = 4;
    const int a_group = 8;
    const int w_group = 3;  // 6 cols -> groups 3+3
    QuantizedMatrix qx = quantize_grouped(xr, a_bits, a_group);
    QuantizedMatrix qw = quantize_grouped(wt, w_bits, w_group);
    for (int32_t code : qx.codes) {
        CHECK(code >= -qmax_for_bits(a_bits));
        CHECK(code <= qmax_for_bits(a_bits));
    }
    for (int32_t code : qw.codes) {
        CHECK(code >= -qmax_for_bits(w_bits));
        CHECK(code <= qmax_for_bits(w_bits));
    }

    Matrix xdq = dequantize_grouped(qx);
    Matrix wdq = dequantize_grouped(qw);
    ErrorStats xerr = quantization_error(xr, xdq);
    ErrorStats werr = quantization_error(wt, wdq);
    CHECK(std::isfinite(xerr.rmse));
    CHECK(std::isfinite(werr.rmse));
    CHECK(xerr.max_abs >= 0.0);

    // Dequantized matrices feed the simulated rotated linear layer.
    Matrix simulated = xdq * wdq;
    Matrix simulated_helper = simulate_quantized_linear(xr, wt, a_bits, w_bits,
                                                        a_group, w_group);
    CHECK(simulated.allFinite());
    CHECK_CLOSE(max_abs_diff(simulated, simulated_helper), 0.0, 1e-5);

    // Bit packing round-trips both tensors exactly.
    std::vector<uint8_t> px = pack_codes(qx);
    std::vector<uint8_t> pw = pack_codes(qw);
    CHECK(unpack_codes(px, a_bits, rows, n) == qx.codes);
    CHECK(unpack_codes(pw, w_bits, n, out) == qw.codes);

    // Persist and reload the calibration parameters.
    const std::string path = "/tmp/oqcb_e2e.bin";
    std::remove(path.c_str());
    save_calibration(path, result, config);
    CalibrationArtifact art = load_calibration(path);
    CHECK_CLOSE(max_abs_diff(art.result.rotation, result.rotation), 0.0, 1e-7);
    CHECK(art.config.seed == config.seed);
    std::remove(path.c_str());
}

TEST_CASE("e2e/two_bit_quantization_runs_and_reports_error") {
    Matrix x(64, 4);
    std::mt19937_64 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 4; ++j) {
            x(i, j) = normal(rng);
        }
    }
    QuantizedMatrix q = quantize_grouped(x, 2, 2);
    Matrix dq = dequantize_grouped(q);
    ErrorStats stats = quantization_error(x, dq);
    CHECK(std::isfinite(stats.rmse));
    // 2 levels cannot represent continuous Gaussian data exactly.
    CHECK(stats.rmse > 0.0);
    CHECK(stats.relative_rmse > 0.0);
}
