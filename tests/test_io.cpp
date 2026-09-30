#include <cstdio>
#include <fstream>

#include "test_util.hpp"

using namespace calibrator;

namespace {

std::string temp_path(const std::string& name) {
    const std::string path = std::string("/tmp/") + name;
    std::remove(path.c_str());
    return path;
}

CalibrationResult make_result(int n) {
    CalibrationResult r;
    r.rotation = Matrix::Identity(n, n);
    r.initial_loss = 0.5;
    r.final_loss = 0.4;
    r.iterations_run = 10;
    r.accepted_steps = 3;
    return r;
}

}  // namespace

TEST_CASE("io/save_load_roundtrip") {
    const std::string path = temp_path("oqcb_roundtrip.bin");
    CalibrationConfig config{256, 424242};
    CalibrationResult result = make_result(8);
    save_calibration(path, result, config);

    CalibrationArtifact art = load_calibration(path);
    CHECK_CLOSE(max_abs_diff(art.result.rotation, result.rotation), 0.0, 1e-7);
    CHECK(art.config.max_iterations == 256);
    CHECK(art.config.seed == 424242u);
    CHECK_CLOSE(art.result.initial_loss, 0.5, 1e-12);
    CHECK_CLOSE(art.result.final_loss, 0.4, 1e-12);
    CHECK(art.result.iterations_run == 10);
    CHECK(art.result.accepted_steps == 3);
    std::remove(path.c_str());
}

TEST_CASE("io/missing_file_is_rejected") {
    bool threw = false;
    try {
        (void)load_calibration("/tmp/definitely_not_present_oqcb_123.bin");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("io/corrupted_byte_is_rejected") {
    const std::string path = temp_path("oqcb_corrupt.bin");
    CalibrationConfig config{16, 1};
    save_calibration(path, make_result(4), config);

    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(10);
    unsigned char flipped = 0xff;
    f.write(reinterpret_cast<const char*>(&flipped), 1);
    f.close();

    bool threw = false;
    try {
        (void)load_calibration(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}

TEST_CASE("io/truncated_file_is_rejected") {
    const std::string path = temp_path("oqcb_trunc.bin");
    CalibrationConfig config{16, 1};
    save_calibration(path, make_result(4), config);

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    auto size = static_cast<size_t>(in.tellg());
    in.seekg(0);
    std::vector<char> data(size);
    in.read(data.data(), static_cast<std::streamsize>(size));
    in.close();
    data.resize(5);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(data.data(), 5);
    out.close();

    bool threw = false;
    try {
        (void)load_calibration(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}

TEST_CASE("io/foreign_magic_is_rejected") {
    const std::string path = temp_path("oqcb_magic.bin");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        std::string junk = "XXXX padding padding padding padding padding";
        out.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    bool threw = false;
    try {
        (void)load_calibration(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    std::remove(path.c_str());
}
