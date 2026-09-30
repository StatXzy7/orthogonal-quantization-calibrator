#include "test_util.hpp"

using namespace calibrator;

namespace {

QuantizedMatrix make_qm(int rows, int cols, int bits, int group_size,
                        std::vector<int32_t> codes) {
    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.bits = bits;
    q.group_size = group_size;
    q.codes = std::move(codes);
    q.scales.assign(static_cast<size_t>(rows) * q.num_groups(), 0.01f);
    return q;
}

}  // namespace

TEST_CASE("pack/exhaustive_2bit_roundtrip") {
    QuantizedMatrix q = make_qm(1, 4, 2, 1, {-1, 0, 1, -1});
    std::vector<uint8_t> packed = pack_codes(q);
    CHECK(packed.size() == 1);
    std::vector<int32_t> back = unpack_codes(packed, 2, 1, 4);
    CHECK(back == q.codes);
}

TEST_CASE("pack/exhaustive_4bit_signed_boundaries") {
    QuantizedMatrix q = make_qm(2, 4, 4, 2, {-8, -7, 0, 7, 7, 0, -8, -1});
    // qmax for 4 bits is 7: -8 is out of the declared range and must be rejected.
    bool threw = false;
    try {
        (void)pack_codes(q);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    QuantizedMatrix ok = make_qm(2, 4, 4, 2, {-7, -1, 0, 7, 7, 0, -7, 1});
    std::vector<uint8_t> packed = pack_codes(ok);
    CHECK(packed.size() == 4);  // 2 bytes per row * 2 rows
    std::vector<int32_t> back = unpack_codes(packed, 4, 2, 4);
    CHECK(back == ok.codes);
}

TEST_CASE("pack/8bit_signed_extremes") {
    QuantizedMatrix q = make_qm(3, 3, 8, 3, {-127, 0, 127, 64, -64, 1, -1, 99, -99});
    std::vector<uint8_t> packed = pack_codes(q);
    CHECK(packed.size() == 9);
    std::vector<int32_t> back = unpack_codes(packed, 8, 3, 3);
    CHECK(back == q.codes);
}

TEST_CASE("pack/tail_group_and_row_padding_roundtrip") {
    // 5 columns, 2-bit fields: 10 bits per row -> 2 bytes per row (6 pad bits).
    std::vector<int32_t> codes = {1, -1, 0, 1, 0, 0, 1, -1, -1, 1};
    QuantizedMatrix q = make_qm(2, 5, 2, 3, codes);
    std::vector<uint8_t> packed = pack_codes(q);
    CHECK(packed.size() == 4);
    std::vector<int32_t> back = unpack_codes(packed, 2, 2, 5);
    CHECK(back == codes);
}

TEST_CASE("pack/random_roundtrip_all_widths") {
    std::mt19937_64 rng(2024);
    for (int bits : {2, 4, 8}) {
        const int qmax = qmax_for_bits(bits);
        std::uniform_int_distribution<int32_t> dist(-qmax, qmax);
        const int rows = 5;
        const int cols = 13;  // awkward width -> tail handling
        std::vector<int32_t> codes(rows * cols);
        for (auto& c : codes) {
            c = dist(rng);
        }
        QuantizedMatrix q = make_qm(rows, cols, bits, 5, codes);
        std::vector<uint8_t> packed = pack_codes(q);
        std::vector<int32_t> back = unpack_codes(packed, bits, rows, cols);
        CHECK(back == codes);
    }
}

TEST_CASE("pack/truncated_buffer_is_rejected") {
    QuantizedMatrix q = make_qm(4, 4, 4, 4, {-7, 7, 0, -1, 1, 0, 7, -7, -1, -1, 1, 1, 7, -7, 0, 0});
    std::vector<uint8_t> packed = pack_codes(q);
    packed.pop_back();
    bool threw = false;
    try {
        (void)unpack_codes(packed, 4, 4, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}
