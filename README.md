# orthogonal-quantization-calibrator

Linux/CPU numerical calibrator (C++17, Eigen only). It mitigates extreme
channel values of a calibration activation matrix by applying an **orthogonal
rotation**, then performs **grouped signed-symmetric low-bit quantization** of
the rotated activations and of the synchronously rotated linear weights.

No model downloads, no external services. The only required third-party
dependency is Eigen (header-only).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Eigen is located via `find_package(Eigen3)` and falls back to searching for
the `Eigen/Dense` header under `/usr/include` / `/usr/local/include`
(e.g. Debian's `/usr/include/eigen3`).

## Mathematical setup

Linear layer (per token/column): `y = W a`, with `W` of shape `(out, d)` and
activation `A` of shape `(d, tokens)`. For an orthogonal matrix `R`
(`R^T R = I`):

- rotated activation: `A' = R A`
- synchronously rotated weight: `W' = W R^T`
- unchanged output: `W' A' = W R^T R A = W A`

The unquantized rotated output is therefore bit-for-bit equivalent (up to
floating-point roundoff) to the original linear output.

### Calibration objective

```
L(R) = mean( exp( - |R A| ) )          # elementwise
```

Each iteration multiplies the current best rotation on the left by a random
Givens (plane) rotation with uniformly chosen plane and an angle uniform on
`[-pi/2, pi/2]`. Orthogonality is therefore preserved by construction at every
iteration. A candidate is accepted only when its loss is strictly smaller and
finite; otherwise the previous valid result is retained. The returned rotation
is the minimum-loss entry among all valid history entries (always including the
initial one at iteration 0).

Inputs: calibration matrix, linear weight, initial rotation, iteration cap,
and a 64-bit random seed. Constant or rank-deficient calibration matrices keep
finite values throughout; no division is performed on activations.

### Quantization

Signed symmetric per-group quantization, independently selectable group sizes
for activations and weights:

- supported bit widths `b in {2, 4, 8}`
- integer levels fixed at `+/- (2^(b-1) - 1)` (b=2: +/-1, b=4: +/-7, b=8: +/-127)
- per-group scale `s = max(|x|) / level` (symmetric, no zero-point)
- code `q = round(clamp(x / s, -level, level))`; dequantize `x_hat = q * s`
- grouping along axis 0 (rows) or axis 1 (columns); the final group is allowed
  to be smaller than `group_size` and is handled correctly
- all-zero groups use scale 0, all-zero codes, and dequantize exactly to zero

This module does **not** promise that quantization error decreases for
arbitrary data; rotation optimizes the stated activation objective only.
Quantization errors (max abs, mean abs, RMSE) are reported via `QuantStats`.

### Bit packing

`pack_codes` / `unpack_codes` perform bit-level packing of signed two's
complement codes for b = 2/4/8, LSB-first. Format: 8-byte magic `OQCPACK1`,
`uint16 bits`, `uint32 count`, `uint32 payload_bytes`, then the bit payload.
Trailing pad bits are permitted; unpack validates magic, bit width, count,
payload length, and the signed code range.

### Calibration archives

`save_calibration` / `load_calibration` use little-endian binary format with
8-byte magic `OQCCALIB`, version, activation/weight matrices, best rotation,
best loss, accepted count, and full history (each entry: iteration, loss,
rotation). `save_quantized` / `load_quantized` (magic `OQCQUANT`) store the
full quantized tensor (shape, grouping, codes, scales). Every file ends in a
FNV-1a 32-bit checksum and is fully validated on load; corruption, dimension
conflicts, or non-finite values throw `oqc::CalibError`.

## Constraints and errors

All matrix dimensions are limited to `16` or smaller. Explicit exceptions
(`oqc::CalibError`, derived from `std::runtime_error`) are raised for:

- non-orthogonal or non-finite initial rotation
- bit width outside {2, 4, 8} or non-positive group size
- dimension conflicts (rotation vs. activations vs. weights, group vs. dim)
- non-finite (NaN/Inf) inputs

## Library entry points

Header: `include/oqc/calibrator.h`

- `calibrate(A, R0, config)` -> `CalibResult { rotation, loss, accepted_iters, history }`
- `rotate_activations`, `rotate_weights`, `rotation_loss`
- `givens_rotation`, `random_rotation`
- `quantize_symmetric` -> `QuantTensor { codes, scales, ... }`
- `dequantize`, `quantize_with_stats` -> `QuantStats { max/mean/rmse, dequantized }`
- `pack_codes`, `unpack_codes`
- `save_calibration`, `load_calibration`, `save_quantized`, `load_quantized`

## Tests and tolerances

The test binary uses a small built-in framework (no additional third-party
dependency). CTest registers every `TEST_CASE` separately plus an aggregate
run; zero registered or zero matched cases make the test binary exit
non-zero.

Assertions cover: orthogonality of every accepted rotation
(`||R^T R - I||_max <= 1e-9`), unquantized rotated output equivalence
(`<= 1e-9 * max(1, ||y||_max)`), signed integer ranges for b=2/4/8,
bit-pack round trips, all-zero groups, short tail groups, axis-1 grouping,
constant/rank-deficient/all-zero inputs, seed determinism, no-improvement
retention, 1x1 and 16x16 boundaries, corrupt archives, and explicit error
paths. Archive comparisons use tolerance `1e-12`.
