#include "oqc/calibrator.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace oqc {

void validate_finite(const Matrix& m, const char* name) {
    if (m.rows() == 0 || m.cols() == 0) {
        throw CalibError(std::string(name) + " must be non-empty");
    }
    if (m.rows() > kMaxDim || m.cols() > kMaxDim) {
        throw CalibError(std::string(name) + " dimensions must be <= " +
                         std::to_string(kMaxDim));
    }
    if (!m.allFinite()) {
        throw CalibError(std::string(name) + " contains non-finite values");
    }
}

void validate_rotation(const Matrix& r) {
    if (r.rows() != r.cols()) {
        throw CalibError("rotation matrix must be square");
    }
    if (r.rows() == 0 || r.rows() > kMaxDim) {
        throw CalibError("rotation dimension must be in [1, " +
                         std::to_string(kMaxDim) + "]");
    }
    if (!r.allFinite()) {
        throw CalibError("rotation contains non-finite values");
    }
    const double tol = 1e-9 * std::max(1.0, (double)r.rows());
    Matrix eye = Matrix::Identity(r.rows(), r.rows());
    Matrix rr = r.transpose() * r;
    double ortho_err = (rr - eye).cwiseAbs().maxCoeff();
    if (ortho_err > tol) {
        throw CalibError("initial rotation is not orthogonal (R^T R - I max = " +
                         std::to_string(ortho_err) + ")");
    }
}

void validate_quant_config(const QuantConfig& qc, int dim) {
    if (qc.bits != 2 && qc.bits != 4 && qc.bits != 8) {
        throw CalibError("bits must be one of {2, 4, 8}");
    }
    if (qc.group_size <= 0) {
        throw CalibError("group_size must be positive");
    }
    if (qc.group_size > kMaxDim) {
        throw CalibError("group_size must be <= " + std::to_string(kMaxDim));
    }
    if (qc.axis != 0 && qc.axis != 1) {
        throw CalibError("quantization axis must be 0 or 1");
    }
    if (dim <= 0 || dim > kMaxDim) {
        throw CalibError("quantized dimension out of range");
    }
    if (qc.group_size > dim) {
        throw CalibError("group_size cannot exceed grouped dimension");
    }
}

Matrix givens_rotation(int n, int p, int q, double angle) {
    if (n < 1 || p < 0 || q < 0 || p >= n || q >= n || p == q) {
        throw CalibError("invalid givens rotation indices");
    }
    if (!std::isfinite(angle)) {
        throw CalibError("givens angle must be finite");
    }
    Matrix g = Matrix::Identity(n, n);
    double c = std::cos(angle);
    double s = std::sin(angle);
    g(p, p) = c;
    g(q, q) = c;
    g(p, q) = -s;
    g(q, p) = s;
    return g;
}

Matrix random_rotation(int n, std::uint64_t seed) {
    if (n < 1 || n > kMaxDim) {
        throw CalibError("random_rotation dimension out of range");
    }
    std::mt19937_64 rng(seed);
    Matrix gaussian(n, n);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            gaussian(i, j) = nd(rng);
        }
    }
    Eigen::HouseholderQR<Matrix> qr(gaussian);
    Matrix q = qr.householderQ();
    Matrix d = (q.transpose() * gaussian).diagonal();
    for (int j = 0; j < n; ++j) {
        if (d(j) < 0.0) {
            q.col(j) = -q.col(j);
        }
    }
    return q;
}

double rotation_loss(const Matrix& activations, const Matrix& rotation) {
    Matrix ra = rotate_activations(activations, rotation);
    if (!ra.allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    const Eigen::ArrayXXd abs_vals = ra.cwiseAbs().array();
    const Eigen::ArrayXXd exponents = -abs_vals;
    return Eigen::exp(exponents).mean();
}

Matrix rotate_activations(const Matrix& activations, const Matrix& rotation) {
    return rotation * activations;
}

Matrix rotate_weights(const Matrix& weights, const Matrix& rotation) {
    return weights * rotation.transpose();
}

CalibResult calibrate(const Matrix& activations,
                      const Matrix& initial_rotation,
                      const CalibConfig& config) {
    validate_finite(activations, "activations");
    validate_rotation(initial_rotation);
    if (initial_rotation.rows() != activations.rows()) {
        throw CalibError("rotation dimension does not match activation rows");
    }
    if (config.max_iters < 0) {
        throw CalibError("max_iters must be non-negative");
    }

    const int n = activations.rows();
    CalibResult result;
    result.rotation = initial_rotation;
    result.loss = rotation_loss(activations, initial_rotation);
    result.accepted_iters = 0;
    if (!std::isfinite(result.loss)) {
        throw CalibError("initial rotation produces non-finite loss");
    }
    result.history.push_back({0, result.loss, result.rotation});

    // In dimension 1 the only orthogonal matrix is [+1] or [-1], and there
    // is no Givens plane to perturb, so nothing can improve the objective.
    if (n == 1) {
        return result;
    }

    std::mt19937_64 rng(config.seed);
    std::uniform_int_distribution<int> pick_plane(0, n - 1);
    std::uniform_real_distribution<double> pick_angle(-M_PI / 2.0, M_PI / 2.0);

    for (int it = 1; it <= config.max_iters; ++it) {
        int p = pick_plane(rng);
        int q = pick_plane(rng);
        while (q == p) {
            q = pick_plane(rng);
        }
        double angle = pick_angle(rng);
        Matrix step = givens_rotation(n, p, q, angle);
        Matrix candidate = step * result.rotation;

        double cand_loss = rotation_loss(activations, candidate);
        if (std::isfinite(cand_loss) && cand_loss < result.loss) {
            result.rotation = candidate;
            result.loss = cand_loss;
            ++result.accepted_iters;
            result.history.push_back({it, cand_loss, candidate});
        }
    }
    return result;
}

double symmetric_scale(const double* vals, int count, int levels) {
    double amax = 0.0;
    for (int i = 0; i < count; ++i) {
        amax = std::max(amax, std::fabs(vals[i]));
    }
    if (amax == 0.0) {
        return 0.0;
    }
    return amax / static_cast<double>(levels);
}

}  // namespace oqc
