#ifndef OQC_CALIBRATOR_H
#define OQC_CALIBRATOR_H

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace oqc {

constexpr int kMaxDim = 16;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct CalibConfig {
    int max_iters = 100;
    std::uint64_t seed = 0xC0FFEEULL;
};

struct HistoryEntry {
    int iter;
    double loss;
    Matrix rotation;
};

struct CalibResult {
    Matrix rotation;
    double loss;
    int accepted_iters;
    std::vector<HistoryEntry> history;
};

struct QuantConfig {
    int bits = 8;
    int group_size = 16;
    int axis = 0;
};

struct QuantTensor {
    int bits = 8;
    int group_size = 16;
    int axis = 0;
    int rows = 0;
    int cols = 0;
    std::vector<std::int32_t> codes;
    std::vector<double> scales;

    const std::int32_t* row_begin(int r) const {
        return codes.data() + static_cast<std::size_t>(r) * cols;
    }
    std::int32_t* row_begin(int r) {
        return codes.data() + static_cast<std::size_t>(r) * cols;
    }
};

struct QuantStats {
    double max_abs_error;
    double mean_abs_error;
    double rmse;
    Matrix dequantized;
};

class CalibError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void validate_finite(const Matrix& m, const char* name);
void validate_rotation(const Matrix& r);
void validate_quant_config(const QuantConfig& qc, int dim);

double rotation_loss(const Matrix& activations, const Matrix& rotation);
CalibResult calibrate(const Matrix& activations,
                      const Matrix& initial_rotation,
                      const CalibConfig& config);

Matrix rotate_activations(const Matrix& activations, const Matrix& rotation);
Matrix rotate_weights(const Matrix& weights, const Matrix& rotation);

Matrix givens_rotation(int n, int p, int q, double angle);
Matrix random_rotation(int n, std::uint64_t seed);

double symmetric_scale(const double* vals, int count, int levels);

QuantTensor quantize_symmetric(const Matrix& m, const QuantConfig& config);
Matrix dequantize(const QuantTensor& q);
QuantStats quantize_with_stats(const Matrix& m, const QuantConfig& config);

std::vector<std::uint8_t> pack_codes(const QuantTensor& q);
std::vector<std::int32_t> unpack_codes(const std::vector<std::uint8_t>& packed,
                                       int bits, std::size_t count);

void save_calibration(const std::string& path,
                      const CalibResult& result,
                      const Matrix& activations,
                      const Matrix& weights);

struct CalibArchive {
    CalibResult result;
    Matrix activations;
    Matrix weights;
};

CalibArchive load_calibration(const std::string& path);

void save_quantized(const std::string& path, const QuantTensor& q);
QuantTensor load_quantized(const std::string& path);

}  // namespace oqc

#endif
