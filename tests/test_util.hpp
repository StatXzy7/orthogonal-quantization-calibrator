#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "calibrator/calibration.hpp"
#include "calibrator/io.hpp"
#include "calibrator/pack.hpp"
#include "calibrator/quantize.hpp"

namespace testing {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int run_all() {
    int failures = 0;
    for (const auto& test : registry()) {
        std::printf("[ RUN      ] %s\n", test.name.c_str());
        std::fflush(stdout);
        try {
            test.fn();
            std::printf("[       OK ] %s\n", test.name.c_str());
        } catch (const std::exception& e) {
            std::printf("[ RUN      ] %s\n", test.name.c_str());
            std::printf("[  FAILED  ] %s: %s\n", test.name.c_str(), e.what());
            ++failures;
        }
    }
    std::printf("[==========] %zu tests registered, %d failure(s)\n",
                registry().size(), failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace testing

#define TEST_CONCAT2(a, b) a##b
#define TEST_CONCAT(a, b) TEST_CONCAT2(a, b)
#define TEST_CASE(name)                                                          \
    static void TEST_CONCAT(test_fn_, __LINE__)();                               \
    static ::testing::Registrar TEST_CONCAT(test_reg_, __LINE__)(                \
        name, &TEST_CONCAT(test_fn_, __LINE__));                                 \
    static void TEST_CONCAT(test_fn_, __LINE__)()

struct AssertionError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::ostringstream oss;                                              \
            oss << __FILE__ << ":" << __LINE__ << " CHECK failed: " #cond;       \
            throw AssertionError(oss.str());                                     \
        }                                                                        \
    } while (0)

#define CHECK_CLOSE(actual, expected, tol)                                       \
    do {                                                                         \
        const double a_ = static_cast<double>(actual);                           \
        const double e_ = static_cast<double>(expected);                         \
        const double d_ = std::fabs(a_ - e_);                                    \
        if (!(d_ <= static_cast<double>(tol))) {                                 \
            std::ostringstream oss;                                              \
            oss << __FILE__ << ":" << __LINE__                                   \
                << " CHECK_CLOSE failed: |" << a_ << " - " << e_ << "| = " << d_ \
                << " > tol " #tol;                                               \
            throw AssertionError(oss.str());                                     \
        }                                                                        \
    } while (0)

inline double max_orthogonality_error(const calibrator::Matrix& r) {
    const calibrator::Matrix eye = calibrator::Matrix::Identity(r.rows(), r.cols());
    return (r.transpose() * r - eye).cwiseAbs().maxCoeff();
}

inline double max_abs_diff(const calibrator::Matrix& a, const calibrator::Matrix& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

inline double relative_fro_diff(const calibrator::Matrix& a,
                                const calibrator::Matrix& b) {
    const double denom = std::max(static_cast<double>(a.norm()), 1e-12);
    return (a - b).norm() / denom;
}

int main() { return ::testing::run_all(); }
