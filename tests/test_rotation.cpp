#include "test_util.hpp"

using namespace calibrator;

namespace {

// Anisotropic activations: one dominant channel, many near-zero channels.
Matrix make_anisotropic(int rows, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix x = Matrix::Zero(rows, n);
    for (int i = 0; i < rows; ++i) {
        x(i, 0) = normal(rng) * 10.0f;
        for (int j = 1; j < n; ++j) {
            x(i, j) = normal(rng) * 0.02f;
        }
    }
    return x;
}

Matrix make_weight(int in, int out, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix w(in, out);
    for (int i = 0; i < in; ++i) {
        for (int j = 0; j < out; ++j) {
            w(i, j) = normal(rng);
        }
    }
    return w;
}

}  // namespace

TEST_CASE("rotation/anisotropic_input_is_improved_and_orthogonal") {
    Matrix x = make_anisotropic(512, 8, 42);
    Matrix w = make_weight(8, 6, 7);
    Matrix r0 = Matrix::Identity(8, 8);
    CalibrationConfig config;
    config.max_iterations = 64;
    config.seed = 1234;

    CalibrationResult result = calibrate(x, w, r0, config);

    CHECK(result.rotation.allFinite());
    CHECK(max_orthogonality_error(result.rotation) < 1e-5);
    CHECK(result.final_loss <= result.initial_loss + 1e-12);
    // Anisotropic data must be balanced across channels by rotation.
    CHECK(result.initial_loss - result.final_loss > 1e-3);
    CHECK(result.accepted_steps > 0);

    Matrix xr = rotate_activations(x, result.rotation);
    Matrix wt = transform_weight(w, result.rotation);
    Matrix original_output = x * w;
    Matrix rotated_output = xr * wt;
    CHECK(relative_fro_diff(original_output, rotated_output) < 1e-4);
}

TEST_CASE("rotation/heavy_tailed_dense_input") {
    const int n = 12;
    std::mt19937_64 rng(99);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix x(300, n);
    for (int i = 0; i < 300; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = normal(rng);
            x(i, j) = (i % 37 == 0) ? v * 20.0f : v * 0.05f;
        }
    }
    Matrix w = make_weight(n, 5, 13);
    Matrix r0 = Matrix::Identity(n, n);
    CalibrationConfig config{128, 55};

    CalibrationResult result = calibrate(x, w, r0, config);

    CHECK(result.rotation.allFinite());
    CHECK(max_orthogonality_error(result.rotation) < 1e-5);
    CHECK(result.final_loss <= result.initial_loss + 1e-12);
    Matrix xr = x * result.rotation;
    Matrix wt = result.rotation.transpose() * w;
    CHECK(relative_fro_diff(x * w, xr * wt) < 1e-4);
}

TEST_CASE("rotation/non_identity_orthogonal_seed_is_accepted") {
    Matrix x = make_anisotropic(128, 4, 1);
    Matrix w = make_weight(4, 3, 2);
    // A real 45-degree rotation in the first plane.
    Matrix r0 = Matrix::Identity(4, 4);
    const float c = std::cos(0.78539816339f);
    const float s = std::sin(0.78539816339f);
    r0(0, 0) = c;
    r0(0, 1) = s;
    r0(1, 0) = -s;
    r0(1, 1) = c;
    CHECK(is_orthogonal(r0));

    CalibrationConfig config{32, 5};
    CalibrationResult result = calibrate(x, w, r0, config);
    CHECK(max_orthogonality_error(result.rotation) < 1e-5);
    CHECK(result.final_loss <= result.initial_loss + 1e-12);
}

TEST_CASE("rotation/zero_iterations_returns_seed_unchanged") {
    Matrix x = make_anisotropic(64, 4, 3);
    Matrix w = make_weight(4, 2, 4);
    Matrix r0 = Matrix::Identity(4, 4);
    CalibrationConfig config{0, 5};

    CalibrationResult result = calibrate(x, w, r0, config);
    CHECK_CLOSE(max_abs_diff(result.rotation, r0), 0.0, 1e-7);
    CHECK_CLOSE(result.final_loss, result.initial_loss, 1e-12);
    CHECK(result.iterations_run == 0);
    CHECK(result.accepted_steps == 0);
}

TEST_CASE("rotation/deterministic_for_same_seed") {
    Matrix x = make_anisotropic(200, 8, 21);
    Matrix w = make_weight(8, 4, 22);
    CalibrationConfig config{64, 777};
    Matrix r0 = Matrix::Identity(8, 8);
    CalibrationResult a = calibrate(x, w, r0, config);
    CalibrationResult b = calibrate(x, w, r0, config);
    CHECK_CLOSE(max_abs_diff(a.rotation, b.rotation), 0.0, 0.0);
    CHECK_CLOSE(a.final_loss, b.final_loss, 0.0);
}
