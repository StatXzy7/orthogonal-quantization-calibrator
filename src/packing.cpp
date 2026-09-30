#include "oqc/calibrator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace oqc {

namespace {

constexpr std::size_t kPackHeaderSize = 18;
constexpr char kPackMagic[8] = {'O', 'Q', 'C', 'P', 'A', 'C', 'K', '1'};

void put_u16le(std::vector<uint8_t>& b, std::size_t pos, uint16_t v) {
    b[pos] = static_cast<uint8_t>(v);
    b[pos + 1] = static_cast<uint8_t>(v >> 8);
}

void put_u32le(std::vector<uint8_t>& b, std::size_t pos, uint32_t v) {
    b[pos] = static_cast<uint8_t>(v);
    b[pos + 1] = static_cast<uint8_t>(v >> 8);
    b[pos + 2] = static_cast<uint8_t>(v >> 16);
    b[pos + 3] = static_cast<uint8_t>(v >> 24);
}

uint16_t get_u16le(const std::vector<uint8_t>& b, std::size_t pos) {
    return static_cast<uint16_t>(b[pos]) |
           (static_cast<uint16_t>(b[pos + 1]) << 8);
}

uint32_t get_u32le(const std::vector<uint8_t>& b, std::size_t pos) {
    return static_cast<uint32_t>(b[pos]) |
           (static_cast<uint32_t>(b[pos + 1]) << 8) |
           (static_cast<uint32_t>(b[pos + 2]) << 16) |
           (static_cast<uint32_t>(b[pos + 3]) << 24);
}

void require(bool cond, const char* msg) {
    if (!cond) {
        throw CalibError(msg);
    }
}

}  // namespace

std::vector<uint8_t> pack_codes(const QuantTensor& q) {
    if (q.bits != 2 && q.bits != 4 && q.bits != 8) {
        throw CalibError("bits must be one of {2,4,8}");
    }
    const int levels = (1 << (q.bits - 1)) - 1;
    for (std::int32_t c : q.codes) {
        if (c < -levels || c > levels) {
            throw CalibError("code outside range cannot be packed");
        }
    }

    const std::size_t payload_bits =
        static_cast<std::size_t>(q.codes.size()) * q.bits;
    const std::size_t payload_bytes = (payload_bits + 7) / 8;
    std::vector<uint8_t> buf(kPackHeaderSize + payload_bytes, 0);
    std::copy(kPackMagic, kPackMagic + 8, buf.begin());
    put_u16le(buf, 8, static_cast<uint16_t>(q.bits));
    put_u32le(buf, 10, static_cast<uint32_t>(q.codes.size()));
    put_u32le(buf, 14, static_cast<uint32_t>(payload_bytes));

    std::size_t bitpos = 0;
    for (std::int32_t c : q.codes) {
        uint32_t u = static_cast<uint32_t>(static_cast<uint8_t>(c));
        for (int k = 0; k < q.bits; ++k) {
            if ((u >> k) & 1u) {
                buf[kPackHeaderSize + bitpos / 8] |=
                    static_cast<uint8_t>(1u << (bitpos % 8));
            }
            ++bitpos;
        }
    }
    return buf;
}

std::vector<std::int32_t> unpack_codes(const std::vector<uint8_t>& packed,
                                       int bits, std::size_t count) {
    if (bits != 2 && bits != 4 && bits != 8) {
        throw CalibError("bits must be one of {2,4,8}");
    }
    require(packed.size() >= kPackHeaderSize,
            "packed buffer too small for header");
    require(std::equal(kPackMagic, kPackMagic + 8, packed.begin()),
            "bad packed-code magic");
    int hdr_bits = get_u16le(packed, 8);
    std::size_t hdr_count = get_u32le(packed, 10);
    std::size_t hdr_payload = get_u32le(packed, 14);
    require(hdr_bits == bits, "packed bit width mismatch");
    require(hdr_count == count, "packed code count mismatch");
    std::size_t need_bytes = (count * bits + 7) / 8;
    require(hdr_payload == need_bytes, "packed payload length mismatch");
    require(packed.size() == kPackHeaderSize + need_bytes,
            "packed buffer size mismatch");

    const int levels = (1 << (bits - 1)) - 1;
    std::vector<std::int32_t> codes(count, 0);
    std::size_t bitpos = 0;
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t u = 0;
        for (int k = 0; k < bits; ++k) {
            std::size_t idx = kPackHeaderSize + bitpos / 8;
            if ((packed[idx] >> (bitpos % 8)) & 1u) {
                u |= 1u << k;
            }
            ++bitpos;
        }
        if (u & (1u << (bits - 1))) {
            u |= ~((1u << bits) - 1u);
        }
        std::int32_t code = static_cast<std::int32_t>(u);
        if (code < -levels || code > levels) {
            throw CalibError("unpacked code outside signed range");
        }
        codes[i] = code;
    }
    return codes;
}

}  // namespace oqc
