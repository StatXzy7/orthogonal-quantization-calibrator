#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "oqc/calibrator.h"

using namespace oqc;

namespace {

constexpr double kEps = 1e-9;

Matrix make_extreme_activations() {
    Matrix a(4, 5);
    a << 10.0, -9.0, 8.0, -7.0, 6.0,
         0.01, -0.02, 0.015, 0.02, -0.01,
         0.1, 0.2, -0.15, 0.1, 0.05,
         -0.3, 0.25, 0.2, -0.1, 0.05;
    return a;
}

Matrix make_weights(int out_features, int in_features) {
    Matrix w(out_features, in_features);
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            w(i, j) = std::sin(0.37 * (i + 1) * (j + 2)) *
                      (1.0 + 0.1 * (i - j));
        }
    }
    return w;
}

double max_orthogonality_error(const Matrix& r) {
    Matrix eye = Matrix::Identity(r.rows(), r.rows());
    return (r.transpose() * r - eye).cwiseAbs().maxCoeff();
}

double max_matrix_diff(const Matrix& a, const Matrix& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

std::string tmp_path(const char* name) {
    const char* tmpdir = std::getenv("TMPDIR");
    std::string base = tmpdir ? tmpdir : "/tmp";
    return base + "/" + name + ".oqc";
}

void assert_codes_in_range(const QuantTensor& q) {
    int levels = (1 << (q.bits - 1)) - 1;
    for (std::int32_t c : q.codes) {
        CHECK(c >= -levels);
        CHECK(c <= levels);
    }
}

}  // namespace

TEST_CASE(calibration_returns_orthogonal_rotation) {
    Matrix a = make_extreme_activations();
    CalibConfig cfg;
    cfg.max_iters = 300;
    cfg.seed = 42;
    CalibResult res = calibrate(a, Matrix::Identity(4, 4), cfg);
    CHECK(res.rotation.rows() == 4);
    CHECK_CLOSE(max_orthogonality_error(res.rotation), 0.0, kEps);
    for (const auto& e : res.history) {
        CHECK_CLOSE(max_orthogonality_error(e.rotation), 0.0, kEps);
    }
}

TEST_CASE(calibration_reduces_or_keeps_objective) {
    Matrix a = make_extreme_activations();
    Matrix r0 = Matrix::Identity(4, 4);
    double initial = rotation_loss(a, r0);
    CalibConfig cfg;
    cfg.max_iters = 500;
    cfg.seed = 7;
    CalibResult res = calibrate(a, r0, cfg);
    CHECK(res.loss <= initial + 1e-15);
    double hist_min = std::numeric_limits<double>::infinity();
    for (const auto& e : res.history) {
        hist_min = std::min(hist_min, e.loss);
    }
    CHECK_CLOSE(res.loss, hist_min, 1e-12);
    CHECK(res.loss < initial);
    for (std::size_t i = 1; i < res.history.size(); ++i) {
        CHECK(res.history[i].loss <=
              res.history[i - 1].loss + 1e-15);
    }
}

TEST_CASE(calibration_seed_is_deterministic) {
    Matrix a = make_extreme_activations();
    CalibConfig cfg;
    cfg.max_iters = 200;
    cfg.seed = 123456;
    CalibResult r1 = calibrate(a, Matrix::Identity(4, 4), cfg);
    CalibResult r2 = calibrate(a, Matrix::Identity(4, 4), cfg);
    CHECK_CLOSE(r1.loss, r2.loss, 1e-15);
    CHECK(max_matrix_diff(r1.rotation, r2.rotation) < 1e-15);
    CHECK(r1.accepted_iters == r2.accepted_iters);
}

TEST_CASE(no_improvement_preserves_initial_rotation) {
    // Zero activations make mean(exp(-|R*A|)) identically 1 for every
    // orthogonal R, so no strict improvement is ever possible.
    Matrix a = Matrix::Zero(5, 7);
    Matrix r0 = random_rotation(5, 999);
    CHECK(max_matrix_diff(r0, Matrix::Identity(5, 5)) > 1e-6);
    CalibConfig cfg;
    cfg.max_iters = 200;
    cfg.seed = 11;
    CalibResult res = calibrate(a, r0, cfg);
    CHECK_CLOSE(res.loss, rotation_loss(a, r0), 1e-14);
    CHECK_CLOSE(res.loss, 1.0, 1e-15);
    CHECK(max_matrix_diff(res.rotation, r0) < 1e-14);
    CHECK(res.accepted_iters == 0);
    CHECK(res.history.size() == 1);
}

TEST_CASE(random_rotation_is_orthogonal) {
    for (std::uint64_t seed : {1ULL, 42ULL, 987654321ULL}) {
        for (int n = 1; n <= 16; ++n) {
            Matrix r = random_rotation(n, seed + n);
            CHECK_CLOSE(max_orthogonality_error(r), 0.0, kEps);
        }
    }
}

TEST_CASE(unquantized_rotated_linear_output_equivalent) {
    Matrix a = make_extreme_activations();
    Matrix w = make_weights(3, 4);
    Matrix r0 = random_rotation(4, 2024);
    CalibConfig cfg;
    cfg.max_iters = 120;
    cfg.seed = 555;
    CalibResult res = calibrate(a, r0, cfg);
    Matrix ra = rotate_activations(a, res.rotation);
    Matrix rw = rotate_weights(w, res.rotation);
    Matrix y0 = w * a;
    Matrix y1 = rw * ra;
    CHECK(max_orthogonality_error(res.rotation) < kEps);
    double scale = y0.cwiseAbs().maxCoeff();
    CHECK(max_matrix_diff(y1, y0) <= kEps * std::max(1.0, scale));
    CHECK_CLOSE(max_matrix_diff(rw * (res.rotation * a), y0), 0.0, kEps);
}

TEST_CASE(quantize_integer_ranges_and_stats) {
    Matrix m(4, 3);
    m << -3.0, 2.5, 0.0,
         1.7, -0.2, 3.0,
         0.001, -2.9, 1.1,
         -1.1, 0.9, -0.001;
    for (int bits : {2, 4, 8}) {
        QuantConfig qc;
        qc.bits = bits;
        qc.group_size = 4;
        qc.axis = 0;
        QuantStats st = quantize_with_stats(m, qc);
        assert_codes_in_range(quantize_symmetric(m, qc));
        CHECK(st.max_abs_error >= 0.0);
        CHECK(st.mean_abs_error >= 0.0);
        CHECK(st.rmse >= 0.0);
        CHECK(st.max_abs_error + 1e-12 >= st.mean_abs_error);
        CHECK(st.dequantized.rows() == 4);
        CHECK(st.dequantized.cols() == 3);
        int levels = (1 << (bits - 1)) - 1;
        CHECK(!st.dequantized.isZero(1e-12) || m.isZero());
        (void)levels;
    }
}

TEST_CASE(quantize_extreme_hits_level_for_2_4_8_bits) {
    Matrix m(4, 1);
    m << 1.0, -0.5, 0.25, -1.0;
    for (int bits : {2, 4, 8}) {
        QuantConfig qc;
        qc.bits = bits;
        qc.group_size = 4;
        QuantTensor q = quantize_symmetric(m, qc);
        int levels = (1 << (bits - 1)) - 1;
        CHECK(q.codes[0] == levels);
        CHECK(q.codes[3] == -levels);
    }
}

TEST_CASE(zero_groups_dequantize_to_zero) {
    Matrix m(6, 2);
    m << 1.0, 0.0,
         0.5, 0.0,
        -0.25, 0.0,
         0.0, 0.0,
         0.0, 0.0,
         0.0, 0.0;
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 3;
    qc.axis = 0;
    QuantTensor q = quantize_symmetric(m, qc);
    Matrix dq = dequantize(q);
    CHECK(dq.col(1).isZero(0.0));
    for (int r = 3; r < 6; ++r) {
        CHECK(std::fabs(dq(r, 0)) == 0.0);
    }
    CHECK(std::fabs(q.scales[1 * 2 + 0]) == 0.0);
    CHECK(std::fabs(q.scales[0 * 2 + 1]) == 0.0);
}

TEST_CASE(tail_group_smaller_than_group_size) {
    Matrix m(5, 1);
    m << 2.0, -1.0, 0.5, 0.25, -2.0;
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 3;
    qc.axis = 0;
    QuantTensor q = quantize_symmetric(m, qc);
    int ng = (5 + 3 - 1) / 3;
    CHECK(ng == 2);
    CHECK(static_cast<int>(q.scales.size()) == ng);
    Matrix dq = dequantize(q);
    Matrix diff = dq - m;
    double scale = 2.0 / 7.0;
    CHECK(diff.cwiseAbs().maxCoeff() <= scale / 2.0 + 1e-12);
    CHECK(q.codes.size() == 5);
    CHECK(q.codes[4] == -7);
}

TEST_CASE(quant_axis_one_grouping_matches) {
    Matrix m(2, 5);
    m << 1.0, 2.0, -2.0, 0.5, 0.0,
         0.0, -1.0, 1.0, -0.25, 0.1;
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 2;
    qc.axis = 1;
    QuantTensor q = quantize_symmetric(m, qc);
    int ng = (5 + 2 - 1) / 2;
    CHECK(ng == 3);
    CHECK(static_cast<int>(q.scales.size()) == 2 * ng);
    Matrix dq = dequantize(q);
    CHECK((dq - m).cwiseAbs().maxCoeff() >= 0.0);
    // axis=1 groups columns: group 0 is columns {0,1}, max abs = 2.0.
    CHECK_CLOSE(dq(0, 0), 4.0 * (2.0 / 7.0), 1e-12);
    // tail group is column {4} alone; row 0 zeroes exactly.
    CHECK_CLOSE(dq(0, 4), 0.0, 1e-12);
    CHECK_CLOSE(q.scales[0 * ng + 0], 2.0 / 7.0, 1e-15);
}

TEST_CASE(pack_unpack_roundtrip_all_bitwidths) {
    Matrix m(7, 3);
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 3; ++c) {
            m(r, c) = std::tanh(0.6 * r - 0.4 * c) * (r + 1);
        }
    }
    for (int bits : {2, 4, 8}) {
        QuantConfig qc;
        qc.bits = bits;
        qc.group_size = 3;
        QuantTensor q = quantize_symmetric(m, qc);
        std::vector<std::uint8_t> packed = pack_codes(q);
        std::size_t payload_bits = q.codes.size() * bits;
        const std::size_t kHeaderBytes = 18;
        CHECK(packed.size() == kHeaderBytes + (payload_bits + 7) / 8);
        std::vector<std::int32_t> got =
            unpack_codes(packed, bits, q.codes.size());
        CHECK(got == q.codes);
    }
}

TEST_CASE(pack_rejects_mismatched_count_or_bits) {
    Matrix m(4, 2);
    m.setRandom();
    QuantConfig qc;
    qc.bits = 2;
    qc.group_size = 4;
    QuantTensor q = quantize_symmetric(m, qc);
    std::vector<std::uint8_t> packed = pack_codes(q);
    CHECK_THROWS(unpack_codes(packed, 4, q.codes.size()));
    CHECK_THROWS(unpack_codes(packed, 2, q.codes.size() - 1));
    packed.pop_back();
    CHECK_THROWS(unpack_codes(packed, 2, q.codes.size()));
}

TEST_CASE(constant_input_stays_finite) {
    Matrix a = Matrix::Constant(5, 7, 1.5);
    CalibConfig cfg;
    cfg.max_iters = 100;
    cfg.seed = 3;
    CalibResult res = calibrate(a, Matrix::Identity(5, 5), cfg);
    CHECK(std::isfinite(res.loss));
    CHECK(res.rotation.allFinite());
    CHECK_CLOSE(max_orthogonality_error(res.rotation), 0.0, kEps);
    QuantConfig qc;
    qc.bits = 2;
    qc.group_size = 5;
    QuantStats st = quantize_with_stats(a, qc);
    CHECK(st.dequantized.allFinite());
    CHECK(std::isfinite(st.rmse));
}

TEST_CASE(rank_deficient_input_stays_finite) {
    Matrix u(4, 2);
    u << 1.0, 0.0,
         0.5, 1.0,
        -1.0, 0.2,
         0.8, -0.6;
    Matrix vt(2, 6);
    vt.setRandom();
    Matrix a = u * vt;
    CalibConfig cfg;
    cfg.max_iters = 150;
    cfg.seed = 88;
    CalibResult res = calibrate(a, Matrix::Identity(4, 4), cfg);
    CHECK(std::isfinite(res.loss));
    CHECK(res.rotation.allFinite());
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 2;
    QuantStats st = quantize_with_stats(a, qc);
    CHECK(st.dequantized.allFinite());
}

TEST_CASE(all_zero_input_quantizes_and_calibrates) {
    Matrix a = Matrix::Zero(4, 8);
    CalibConfig cfg;
    cfg.max_iters = 40;
    cfg.seed = 17;
    CalibResult res = calibrate(a, Matrix::Identity(4, 4), cfg);
    CHECK_CLOSE(res.loss, 1.0, 1e-15);
    CHECK(std::isfinite(res.loss));
    QuantConfig qc;
    qc.bits = 2;
    qc.group_size = 4;
    Matrix dq = dequantize(quantize_symmetric(a, qc));
    CHECK(dq.isZero(0.0));
}

TEST_CASE(invalid_inputs_throw) {
    Matrix a = make_extreme_activations();
    Matrix bad_nonorth(4, 4);
    bad_nonorth = Matrix::Identity(4, 4);
    bad_nonorth(0, 1) = 0.1;
    CalibConfig cfg;
    cfg.max_iters = 10;
    CHECK_THROWS(calibrate(a, bad_nonorth, cfg));

    Matrix bad_size(3, 3);
    bad_size.setIdentity();
    CHECK_THROWS(calibrate(a, bad_size, cfg));

    Matrix nan_a = a;
    nan_a(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(calibrate(nan_a, Matrix::Identity(4, 4), cfg));

    Matrix inf_a = a;
    inf_a(1, 1) = std::numeric_limits<double>::infinity();
    CHECK_THROWS(calibrate(inf_a, Matrix::Identity(4, 4), cfg));

    Matrix big = Matrix::Zero(17, 2);
    CHECK_THROWS(calibrate(big, Matrix::Identity(17, 17), cfg));

    QuantConfig bad_bits;
    bad_bits.bits = 3;
    bad_bits.group_size = 4;
    CHECK_THROWS(quantize_symmetric(a, bad_bits));

    QuantConfig bad_group;
    bad_group.bits = 4;
    bad_group.group_size = 0;
    CHECK_THROWS(quantize_symmetric(a, bad_group));

    QuantConfig group_too_big;
    group_too_big.bits = 4;
    group_too_big.group_size = 5;
    CHECK_THROWS(quantize_symmetric(a, group_too_big));

    cfg.max_iters = -1;
    CHECK_THROWS(calibrate(a, Matrix::Identity(4, 4), cfg));

    Matrix nan_w = make_weights(3, 4);
    nan_w(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CalibResult good = calibrate(a, Matrix::Identity(4, 4),
                                 CalibConfig{10, 1});
    CHECK_THROWS(save_calibration(tmp_path("should_not_exist"),
                                  good, a, nan_w));
}

TEST_CASE(save_load_calibration_roundtrip) {
    Matrix a = make_extreme_activations();
    Matrix w = make_weights(3, 4);
    CalibConfig cfg;
    cfg.max_iters = 80;
    cfg.seed = 2468;
    CalibResult res = calibrate(a, Matrix::Identity(4, 4), cfg);
    std::string path = tmp_path("calib_roundtrip");
    save_calibration(path, res, a, w);
    CalibArchive ar = load_calibration(path);
    CHECK(max_matrix_diff(ar.activations, a) < 1e-15);
    CHECK(max_matrix_diff(ar.weights, w) < 1e-15);
    CHECK(max_matrix_diff(ar.result.rotation, res.rotation) < 1e-15);
    CHECK_CLOSE(ar.result.loss, res.loss, 1e-12);
    CHECK(ar.result.accepted_iters == res.accepted_iters);
    CHECK(ar.result.history.size() == res.history.size());
}

TEST_CASE(corrupted_calibration_archive_rejected) {
    Matrix a = make_extreme_activations();
    Matrix w = make_weights(3, 4);
    CalibResult res = calibrate(a, Matrix::Identity(4, 4),
                                CalibConfig{20, 9});
    std::string path = tmp_path("calib_corrupt");
    save_calibration(path, res, a, w);
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    CHECK(f != nullptr);
    std::fseek(f, 20, SEEK_SET);
    int ch = std::fgetc(f);
    std::fseek(f, 20, SEEK_SET);
    std::fputc(ch ^ 0xFF, f);
    std::fclose(f);
    CHECK_THROWS(load_calibration(path));
}

TEST_CASE(save_load_quantized_roundtrip) {
    Matrix m(6, 4);
    m.setRandom();
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 4;
    qc.axis = 0;
    QuantTensor q = quantize_symmetric(m, qc);
    std::string path = tmp_path("quant_roundtrip");
    save_quantized(path, q);
    QuantTensor loaded = load_quantized(path);
    CHECK(loaded.bits == q.bits);
    CHECK(loaded.group_size == q.group_size);
    CHECK(loaded.axis == q.axis);
    CHECK(loaded.rows == q.rows);
    CHECK(loaded.cols == q.cols);
    CHECK(loaded.codes == q.codes);
    CHECK(loaded.scales.size() == q.scales.size());
    for (std::size_t i = 0; i < q.scales.size(); ++i) {
        CHECK_CLOSE(loaded.scales[i], q.scales[i], 1e-15);
    }
    CHECK(max_matrix_diff(dequantize(loaded), dequantize(q)) < 1e-15);
}

TEST_CASE(save_load_quantized_axis_one_tail_group) {
    Matrix m(3, 5);
    m.setRandom();
    QuantConfig qc;
    qc.bits = 2;
    qc.group_size = 2;
    qc.axis = 1;
    QuantTensor q = quantize_symmetric(m, qc);
    std::string path = tmp_path("quant_axis1");
    save_quantized(path, q);
    QuantTensor loaded = load_quantized(path);
    CHECK(loaded.codes == q.codes);
    CHECK(loaded.scales == q.scales);
    Matrix dq = dequantize(loaded);
    CHECK(dq.allFinite());
}

TEST_CASE(givens_rotation_orthogonal) {
    for (int n : {2, 3, 8, 16}) {
        for (double angle : {0.0, 0.3, -1.1, 1.57, -3.14}) {
            Matrix g = givens_rotation(n, 0, n - 1, angle);
            CHECK_CLOSE(max_orthogonality_error(g), 0.0, kEps);
        }
    }
    CHECK_THROWS(givens_rotation(3, 0, 0, 0.1));
    CHECK_THROWS(givens_rotation(3, -1, 1, 0.1));
}

TEST_CASE(end_to_end_rotate_quantize_reports_error) {
    Matrix a = make_extreme_activations();
    Matrix w = make_weights(3, 4);
    CalibResult res = calibrate(a, Matrix::Identity(4, 4),
                                CalibConfig{100, 31});
    Matrix ra = rotate_activations(a, res.rotation);
    Matrix rw = rotate_weights(w, res.rotation);
    QuantConfig act_qc;
    act_qc.bits = 4;
    act_qc.group_size = 4;
    act_qc.axis = 0;
    QuantConfig w_qc;
    w_qc.bits = 4;
    w_qc.group_size = 4;
    w_qc.axis = 1;
    QuantStats act_st = quantize_with_stats(ra, act_qc);
    QuantStats w_st = quantize_with_stats(rw, w_qc);
    CHECK(act_st.dequantized.allFinite());
    CHECK(w_st.dequantized.allFinite());
    Matrix y_orig = w * a;
    Matrix y_deq = w_st.dequantized * act_st.dequantized;
    double err = (y_deq - y_orig).cwiseAbs().maxCoeff();
    CHECK(std::isfinite(err));
    CHECK(err >= 0.0);
}

TEST_CASE(max_dimension_sixteen_supported) {
    Matrix a = Matrix::Random(16, 12);
    CalibConfig cfg;
    cfg.max_iters = 60;
    cfg.seed = 5;
    CalibResult res = calibrate(a, Matrix::Identity(16, 16), cfg);
    CHECK(res.rotation.rows() == 16);
    Matrix eye = Matrix::Identity(16, 16);
    CHECK((res.rotation.transpose() * res.rotation - eye)
              .cwiseAbs()
              .maxCoeff() < 1e-8);
    QuantConfig qc;
    qc.bits = 2;
    qc.group_size = 16;
    qc.axis = 0;
    QuantTensor q = quantize_symmetric(a, qc);
    Matrix dq = dequantize(q);
    CHECK(dq.allFinite());
}

TEST_CASE(one_by_one_rotation_quant_pack) {
    Matrix a(1, 1);
    a(0, 0) = 2.5;
    CalibResult res = calibrate(a, Matrix::Identity(1, 1),
                                CalibConfig{20, 8});
    CHECK_CLOSE(res.rotation(0, 0), 1.0, 1e-15);
    CHECK(std::isfinite(res.loss));
    QuantConfig qc;
    qc.bits = 4;
    qc.group_size = 1;
    QuantTensor q = quantize_symmetric(a, qc);
    CHECK(q.codes[0] == 7);
    Matrix dq = dequantize(q);
    CHECK_CLOSE(dq(0, 0), 2.5, 1e-12);
    std::vector<std::uint8_t> packed = pack_codes(q);
    CHECK(packed.size() == 18 + 1);
    std::vector<std::int32_t> got = unpack_codes(packed, 4, 1);
    CHECK(got[0] == 7);
}

TEST_CASE(weight_dimension_conflict_throws) {
    Matrix a = make_extreme_activations();
    Matrix w_bad = make_weights(3, 5);
    CalibResult res = calibrate(a, Matrix::Identity(4, 4),
                                CalibConfig{10, 1});
    CHECK_THROWS(save_calibration(tmp_path("dim_conflict"),
                                  res, a, w_bad));
    Matrix w_ok = make_weights(2, 4);
    Matrix ra = rotate_activations(a, res.rotation);
    Matrix rw = rotate_weights(w_ok, res.rotation);
    CHECK((rw * ra - w_ok * a).cwiseAbs().maxCoeff() < kEps * 10);
}

int main(int argc, char** argv) {
    auto& cases = oqctest::registry();
    if (cases.empty()) {
        std::fprintf(stderr, "FAIL: no test cases registered\n");
        return 2;
    }
    std::string filter = argc > 1 ? argv[1] : "";
    int run = 0;
    int failed = 0;
    for (const auto& tc : cases) {
        if (!filter.empty() && tc.name != filter) {
            continue;
        }
        ++run;
        try {
            tc.fn();
            std::printf("[ PASS ] %s\n", tc.name.c_str());
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[ FAIL ] %s: %s\n", tc.name.c_str(), e.what());
        }
    }
    if (run == 0) {
        std::fprintf(stderr, "FAIL: no test cases matched '%s'\n",
                     filter.c_str());
        return 2;
    }
    std::printf("\n%d/%d passed; tolerances: rotation orthogonality %.0e, "
                "unquantized output %.0e, archive compare %.0e\n",
                run - failed, run, kEps, kEps, 1e-12);
    return failed == 0 ? 0 : 1;
}
