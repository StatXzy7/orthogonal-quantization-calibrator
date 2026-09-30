#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

#include "calibrator/calibration.hpp"
#include "calibrator/io.hpp"
#include "calibrator/pack.hpp"
#include "calibrator/quantize.hpp"

namespace {

void print_stats(const char* name, const calibrator::ErrorStats& s) {
    std::printf("  %-22s rmse=%.6f  mean_abs=%.6f  max_abs=%.6f  rel_rmse=%.6f\n",
                name, s.rmse, s.mean_abs, s.max_abs, s.relative_rmse);
}

}


int main(int argc, char** argv) {
    const int n = 8;
    const int out = 6;
    const int rows = 1024;
    calibrator::CalibrationConfig config;
    config.max_iterations = 128;
    config.seed = (argc > 1) ? std::stoull(argv[1]) : 42ULL;
    const std::string artifact =
        (argc > 2) ? std::string(argv[2]) : std::string("/tmp/calibration_demo.bin");

    std::mt19937_64 rng(config.seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    calibrator::Matrix x(rows, n);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            const float scale = (j == 0) ? 30.0f : (j == 1) ? 0.01f : 0.25f;
            x(i, j) = normal(rng) * scale;
        }
    }
    calibrator::Matrix w(n, out);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < out; ++j) {
            w(i, j) = normal(rng);
        }
    }

    calibrator::Matrix r0 = calibrator::Matrix::Identity(n, n);
    calibrator::CalibrationResult result = calibrator::calibrate(x, w, r0, config);

    std::printf("calibration: loss %.6f -> %.6f, accepted %d / %d sweeps\n",
                result.initial_loss, result.final_loss, result.accepted_steps,
                result.iterations_run);

    calibrator::Matrix xr = calibrator::rotate_activations(x, result.rotation);
    calibrator::Matrix wt = calibrator::transform_weight(w, result.rotation);
    const double equivalence =
        (x * w - xr * wt).norm() / std::max((x * w).norm(), 1e-12f);
    std::printf("unquantized rotated-output relative Frobenius error: %.3e\n",
                equivalence);

    const int a_bits = 4;
    const int w_bits = 4;
    calibrator::QuantizedMatrix qx = calibrator::quantize_grouped(xr, a_bits, 8);
    calibrator::QuantizedMatrix qw = calibrator::quantize_grouped(wt, w_bits, 3);
    calibrator::Matrix xdq = calibrator::dequantize_grouped(qx);
    calibrator::Matrix wdq = calibrator::dequantize_grouped(qw);

    std::printf("quantization (W4A4, activation group 8, weight group 3):\n");
    print_stats("activations", calibrator::quantization_error(xr, xdq));
    print_stats("weights", calibrator::quantization_error(wt, wdq));
    print_stats("linear output", calibrator::quantization_error(x * w, xdq * wdq));

    auto px = calibrator::pack_codes(qx);
    auto pw = calibrator::pack_codes(qw);
    const auto ux = calibrator::unpack_codes(px, a_bits, rows, n);
    const auto uw = calibrator::unpack_codes(pw, w_bits, n, out);
    std::printf("packed activation bytes=%zu, weight bytes=%zu, roundtrip %s\n",
                px.size(), pw.size(), (ux == qx.codes && uw == qw.codes) ? "OK" : "FAIL");

    try {
        calibrator::save_calibration(artifact, result, config);
        auto reloaded = calibrator::load_calibration(artifact);
        const double diff =
            (reloaded.result.rotation - result.rotation).cwiseAbs().maxCoeff();
        std::printf("saved/loaded artifact %s (max rotation diff %.3e)\n",
                    artifact.c_str(), diff);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "artifact io failed: %s\n", e.what());
        return 2;
    }
    return 0;
}
