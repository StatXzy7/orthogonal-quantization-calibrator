#pragma once

#include <string>

#include "calibrator/calibration.hpp"

namespace calibrator {

// Persist calibration parameters (rotation, config, losses, counters) to a
// self-describing little-endian binary file. Weight is not part of the
// artifact.
void save_calibration(const std::string& path, const CalibrationResult& result,
                      const CalibrationConfig& config);

// Load a calibration artifact written by save_calibration. Throws
// std::runtime_error with an explicit message when the file is missing,
// truncated, has an unknown format / version, a mismatched shape or a bad
// checksum.
struct CalibrationArtifact {
    CalibrationResult result;
    CalibrationConfig config;
};
CalibrationArtifact load_calibration(const std::string& path);

}  // namespace calibrator
