#include "calibrator/calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace calibrator {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kGridPoints = 48;  // global grid on [0, pi)
constexpr int kRefineIters = 6;  // golden-section-style local refinements

double pair_sum(const Matrix& x, int p, int q, double c, double s) {
    double sum = 0.0;
    const int rows = x.rows();
    for (int i = 0; i < rows; ++i) {
        const double a = static_cast<double>(x(i, p));
        const double b = static_cast<double>(x(i, q));
        const double u = c * a + s * b;
        const double v = -s * a + c * b;
        sum += std::exp(-std::fabs(u)) + std::exp(-std::fabs(v));
    }
    return sum;
}

void apply_pair(Matrix& x, int p, int q, float c, float s) {
    const int rows = x.rows();
    for (int i = 0; i < rows; ++i) {
        const float a = x(i, p);
        const float b = x(i, q);
        x(i, p) = c * a + s * b;
        x(i, q) = -s * a + c * b;
    }
}

// Two-pass modified Gram-Schmidt orthonormalization of the columns of R.
void reorthonormalize(Matrix& r) {
    const int n = r.rows();
    for (int pass = 0; pass < 2; ++pass) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < j; ++k) {
                const float dot = r.col(j).dot(r.col(k));
                r.col(j) -= dot * r.col(k);
            }
            const float norm = r.col(j).norm();
            if (norm > 0.0f) {
                r.col(j) /= norm;
            }
        }
    }
}

}  // namespace

double calibration_loss(MatrixRef activations) {
    if (activations.size() == 0 || !activations.allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    const double n = static_cast<double>(activations.size());
    double sum_sq = 0.0;
    for (int j = 0; j < activations.cols(); ++j) {
        for (int i = 0; i < activations.rows(); ++i) {
            const double v = static_cast<double>(activations(i, j));
            sum_sq += v * v;
        }
    }
    const double scale = std::sqrt(std::max(sum_sq / n, 1e-300));
    double loss = 0.0;
    for (int j = 0; j < activations.cols(); ++j) {
        for (int i = 0; i < activations.rows(); ++i) {
            loss += std::exp(-std::fabs(static_cast<double>(activations(i, j))) / scale);
        }
    }
    return loss / n;
}

Matrix rotate_activations(MatrixRef activations, MatrixRef rotation) {
    return activations * rotation;
}

Matrix transform_weight(MatrixRef weight, MatrixRef rotation) {
    return rotation.transpose() * weight;
}

bool is_orthogonal(MatrixRef rotation, float tolerance) {
    if (rotation.rows() != rotation.cols() || !rotation.allFinite()) {
        return false;
    }
    const Matrix eye = Matrix::Identity(rotation.rows(), rotation.cols());
    return (rotation.transpose() * rotation - eye).cwiseAbs().maxCoeff() <= tolerance;
}

std::string validate_calibration_inputs(MatrixRef activations, MatrixRef weight,
                                        MatrixRef initial_rotation,
                                        const CalibrationConfig& config) {
    if (activations.rows() <= 0 || activations.cols() <= 0) {
        return "activation matrix must be non-empty";
    }
    if (activations.cols() > kMaxDim) {
        return "activation channel dimension must be <= 16";
    }
    if (activations.rows() > kMaxRows) {
        return "number of activation rows must be <= 65536";
    }
    if (!activations.allFinite()) {
        return "activation matrix contains non-finite values";
    }
    if (weight.rows() != activations.cols() || weight.cols() <= 0) {
        return "weight matrix shape must be (activation_channels, out_features)";
    }
    if (weight.cols() > kMaxDim) {
        return "weight output dimension must be <= 16";
    }
    if (!weight.allFinite()) {
        return "weight matrix contains non-finite values";
    }
    const int n = activations.cols();
    if (initial_rotation.rows() != n || initial_rotation.cols() != n) {
        return "initial rotation must be a square matrix matching the channel dimension";
    }
    if (!initial_rotation.allFinite()) {
        return "initial rotation contains non-finite values";
    }
    if (!is_orthogonal(initial_rotation)) {
        return "initial rotation is not orthogonal (R^T R != I within 1e-5)";
    }
    if (config.max_iterations < 0) {
        return "max_iterations must be non-negative";
    }
    return "";
}

CalibrationResult calibrate(MatrixRef activations, MatrixRef weight,
                            MatrixRef initial_rotation,
                            const CalibrationConfig& config) {
    const std::string error =
        validate_calibration_inputs(activations, weight, initial_rotation, config);
    if (!error.empty()) {
        throw std::invalid_argument(error);
    }

    const int n = activations.cols();
    const int rows = activations.rows();
    const double count = static_cast<double>(activations.size());

    Matrix rotated = activations * initial_rotation;
    Matrix rotation = initial_rotation;

    auto current_scale = [&]() {
        double total = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < rows; ++i) {
                const double v = static_cast<double>(rotated(i, j));
                total += v * v;
            }
        }
        return std::sqrt(std::max(total / count, 1e-300));
    };
    auto column_sum = [&](int col, double scale) {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i) {
            sum += std::exp(-std::fabs(static_cast<double>(rotated(i, col))) / scale);
        }
        return sum;
    };
    auto total_sum = [&](double scale) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < rows; ++i) {
                sum += std::exp(-std::fabs(static_cast<double>(rotated(i, j))) / scale);
            }
        }
        return sum;
    };

    const double initial_scale = current_scale();
    double best_sum = total_sum(initial_scale);
    Matrix best_rotation = rotation;
    const double initial_loss = best_sum / count;

    std::mt19937_64 rng(config.seed);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    int iterations_run = 0;
    int accepted = 0;

    for (int it = 0; it < config.max_iterations; ++it) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int oi = 0; oi + 1 < n; ++oi) {
            const int p = order[oi];
            const int q = order[oi + 1];
            const double scale = current_scale();
            const double baseline = column_sum(p, scale) + column_sum(q, scale);

            // Uniform grid over one period [0, pi).
            double best_theta = 0.0;
            double best_pair = baseline;
            for (int g = 0; g < kGridPoints; ++g) {
                const double theta = kPi * static_cast<double>(g) / kGridPoints;
                const double candidate =
                    pair_sum(rotated, p, q, std::cos(theta), std::sin(theta));
                if (candidate < best_pair) {
                    best_pair = candidate;
                    best_theta = theta;
                }
            }

            // Golden-section local refinement around the best grid point.
            double lo = best_theta - kPi / kGridPoints;
            double hi = best_theta + kPi / kGridPoints;
            const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
            double x1 = hi - gr * (hi - lo);
            double x2 = lo + gr * (hi - lo);
            double f1 = pair_sum(rotated, p, q, std::cos(x1), std::sin(x1));
            double f2 = pair_sum(rotated, p, q, std::cos(x2), std::sin(x2));
            for (int refine = 0; refine < kRefineIters; ++refine) {
                if (f1 < f2) {
                    hi = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = hi - gr * (hi - lo);
                    f1 = pair_sum(rotated, p, q, std::cos(x1), std::sin(x1));
                } else {
                    lo = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = lo + gr * (hi - lo);
                    f2 = pair_sum(rotated, p, q, std::cos(x2), std::sin(x2));
                }
            }
            if (f1 < best_pair) {
                best_pair = f1;
                best_theta = x1;
            }
            if (f2 < best_pair) {
                best_pair = f2;
                best_theta = x2;
            }

            // Accept only a genuine improvement; otherwise the previous valid
            // candidate (R0 or best so far) is retained untouched.
            if (!(best_pair + 1e-12 * count < baseline)) {
                continue;
            }

            const float c = static_cast<float>(std::cos(best_theta));
            const float s = static_cast<float>(std::sin(best_theta));
            apply_pair(rotated, p, q, c, s);
            for (int i = 0; i < n; ++i) {
                const float rp = rotation(i, p);
                const float rq = rotation(i, q);
                rotation(i, p) = c * rp + s * rq;
                rotation(i, q) = -s * rp + c * rq;
            }
            reorthonormalize(rotation);

            // Keep the working activations exactly consistent with X * R so
            // float drift cannot accumulate.
            rotated = activations * rotation;
            if (!rotation.allFinite() || !rotated.allFinite()) {
                continue;
            }

            const double candidate_sum = total_sum(current_scale());
            if (candidate_sum + 1e-12 * count < best_sum) {
                best_sum = candidate_sum;
                best_rotation = rotation;
                ++accepted;
            }
        }
        ++iterations_run;
    }

    reorthonormalize(best_rotation);
    const Matrix final_rotated = activations * best_rotation;
    const double final_loss = calibration_loss(final_rotated);

    CalibrationResult result;
    result.rotation = best_rotation;
    result.initial_loss = initial_loss;
    result.final_loss = std::isfinite(final_loss) ? final_loss : initial_loss;
    result.iterations_run = iterations_run;
    result.accepted_steps = accepted;
    return result;
}

}  // namespace calibrator
