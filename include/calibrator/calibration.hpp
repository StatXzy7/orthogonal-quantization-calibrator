#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace calibrator {

using Matrix = Eigen::MatrixXf;
using MatrixRef = Eigen::Ref<const Matrix>;

inline constexpr int kMaxDim = 16;
inline constexpr int kMaxRows = 65536;
inline constexpr float kOrthogonalityTolerance = 1e-5f;

struct CalibrationConfig {
    int max_iterations = 256;
    std::uint64_t seed = 0;
};

struct CalibrationResult {
    Matrix rotation;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    int iterations_run = 0;
    int accepted_steps = 0;
};

// Loss minimized by calibration: mean over every element of exp(-|x| / s),
// where s is the RMS of the calibrated activations (scale invariant).
// Smaller loss means values are spread away from zero (fewer extreme / dead
// channels).
double calibration_loss(MatrixRef activations);

// Apply a rotation to activations: Xr = X * R.
Matrix rotate_activations(MatrixRef activations, MatrixRef rotation);

// Rotate the columns of a linear weight in lock-step so that the linear map
// is unchanged: Y = X * W = (X * R) * (R^T * W).
Matrix transform_weight(MatrixRef weight, MatrixRef rotation);

// True when R is square, finite and satisfies R^T R = I within tolerance.
bool is_orthogonal(MatrixRef rotation, float tolerance = kOrthogonalityTolerance);

// Validate calibration inputs. Returns "" when everything is valid and a
// human-readable English reason otherwise.
std::string validate_calibration_inputs(MatrixRef activations, MatrixRef weight,
                                        MatrixRef initial_rotation,
                                        const CalibrationConfig& config);

// Run the rotation search. Always returns a finite orthogonal rotation, namely
// the best (lowest loss) valid candidate encountered, including R0 itself.
// Throws std::invalid_argument when the inputs are invalid.
CalibrationResult calibrate(MatrixRef activations, MatrixRef weight,
                            MatrixRef initial_rotation,
                            const CalibrationConfig& config);

}  // namespace calibrator
