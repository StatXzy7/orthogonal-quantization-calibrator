#include <limits>

#include "test_util.hpp"

using namespace calibrator;

namespace {

bool throws_invalid_argument(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

TEST_CASE("validation/non_orthogonal_initial_rotation_is_rejected") {
    Matrix x = Matrix::Random(8, 4);
    Matrix w = Matrix::Random(4, 3);
    Matrix bad = Matrix::Identity(4, 4);
    bad(0, 1) = 0.01f;  // R^T R != I
    CHECK(!is_orthogonal(bad));
    CalibrationConfig config{10, 1};
    CHECK(throws_invalid_argument([&] { calibrate(x, w, bad, config); }));
}

TEST_CASE("validation/dimension_conflicts_are_rejected") {
    Matrix x = Matrix::Random(8, 4);
    Matrix w = Matrix::Random(3, 3);  // rows 3 != x.cols() 4
    Matrix r = Matrix::Identity(4, 4);
    CalibrationConfig config{10, 1};
    CHECK(throws_invalid_argument([&] { calibrate(x, w, r, config); }));

    Matrix w_ok = Matrix::Random(4, 2);
    Matrix r_bad_shape = Matrix::Identity(3, 3);
    CHECK(throws_invalid_argument([&] { calibrate(x, w_ok, r_bad_shape, config); }));
}

TEST_CASE("validation/dimensions_above_sixteen_are_rejected") {
    Matrix x = Matrix::Random(4, 17);
    Matrix w = Matrix::Random(17, 3);
    Matrix r = Matrix::Identity(17, 17);
    CalibrationConfig config{1, 0};
    CHECK(throws_invalid_argument([&] { calibrate(x, w, r, config); }));
}

TEST_CASE("validation/non_finite_inputs_are_rejected") {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Matrix r = Matrix::Identity(4, 4);
    CalibrationConfig config{10, 1};

    Matrix x_nan = Matrix::Random(8, 4);
    x_nan(2, 2) = nan;
    Matrix w = Matrix::Random(4, 3);
    CHECK(throws_invalid_argument([&] { calibrate(x_nan, w, r, config); }));

    Matrix x_inf = Matrix::Random(8, 4);
    x_inf(0, 0) = inf;
    CHECK(throws_invalid_argument([&] { calibrate(x_inf, w, r, config); }));

    Matrix w_nan = Matrix::Random(4, 3);
    w_nan(0, 0) = nan;
    Matrix x_ok = Matrix::Random(8, 4);
    CHECK(throws_invalid_argument([&] { calibrate(x_ok, w_nan, r, config); }));

    Matrix r_nan = Matrix::Identity(4, 4);
    r_nan(0, 0) = nan;
    CHECK(throws_invalid_argument([&] { calibrate(x_ok, w, r_nan, config); }));

    // Quantization also rejects non-finite values.
    Matrix q_nan = Matrix::Random(2, 2);
    q_nan(0, 0) = nan;
    CHECK(throws_invalid_argument([&] { quantize_grouped(q_nan, 4, 2); }));
}

TEST_CASE("validation/illegal_bit_width_and_group_size") {
    Matrix m = Matrix::Random(4, 4);
    CHECK(throws_invalid_argument([&] { quantize_grouped(m, 3, 2); }));
    CHECK(throws_invalid_argument([&] { quantize_grouped(m, 1, 2); }));
    CHECK(throws_invalid_argument([&] { quantize_grouped(m, 16, 2); }));
    CHECK(throws_invalid_argument([&] { quantize_grouped(m, 4, 0); }));
    QuantizedMatrix bad;
    bad.rows = 1;
    bad.cols = 2;
    bad.bits = 3;
    bad.group_size = 1;
    bad.codes = {0, 0};
    bad.scales = {0.0f};
    CHECK(throws_invalid_argument([&] { dequantize_grouped(bad); }));
}

TEST_CASE("validation/constant_and_rank_deficient_inputs_stay_finite") {
    CalibrationConfig config{32, 9};
    Matrix r = Matrix::Identity(4, 4);

    Matrix constant = Matrix::Constant(32, 4, 3.5f);
    Matrix w = Matrix::Random(4, 2);
    CalibrationResult a = calibrate(constant, w, r, config);
    CHECK(a.rotation.allFinite());
    CHECK(std::isfinite(a.initial_loss));
    CHECK(std::isfinite(a.final_loss));
    CHECK(max_orthogonality_error(a.rotation) < 1e-5);

    Matrix rank1 = Matrix::Zero(40, 4);
    std::mt19937_64 rng(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Eigen::VectorXf col(40), row(4);
    for (int i = 0; i < 40; ++i) {
        col(i) = normal(rng);
    }
    for (int j = 0; j < 4; ++j) {
        row(j) = normal(rng) * 1e-6f;
    }
    rank1 = col * row.transpose();
    CalibrationResult b = calibrate(rank1, w, r, config);
    CHECK(b.rotation.allFinite());
    CHECK(std::isfinite(b.final_loss));
    CHECK(max_orthogonality_error(b.rotation) < 1e-5);

    Matrix all_zero = Matrix::Zero(16, 4);
    CalibrationResult c = calibrate(all_zero, w, r, config);
    CHECK(c.rotation.allFinite());
    CHECK(std::isfinite(c.initial_loss));
    CHECK(std::isfinite(c.final_loss));
    CHECK(c.final_loss <= c.initial_loss + 1e-12);
}

TEST_CASE("validation/single_channel_case") {
    Matrix x = Matrix::Random(16, 1);
    Matrix w = Matrix::Random(1, 1);
    Matrix r = Matrix::Identity(1, 1);
    CalibrationConfig config{8, 1};
    CalibrationResult result = calibrate(x, w, r, config);
    CHECK(result.rotation.size() == 1);
    CHECK_CLOSE(std::fabs(result.rotation(0, 0)), 1.0, 1e-6);
    CHECK(max_orthogonality_error(result.rotation) < 1e-5);
}
