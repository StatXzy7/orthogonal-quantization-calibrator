# 正交旋转数值校准器（Linux / CPU / Eigen）

用正交旋转缓解矩阵激活的极端通道值，并对旋转后的激活与同步变换的权重做分组
低位（2/4/8 bit）有符号对称量化。仅依赖 Eigen，无网络、无模型下载、无外部服务。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/calibrator_demo [seed] [artifact_path]
```

需要 g++（C++17）、CMake ≥ 3.16、系统 Eigen3（`find_package(Eigen3)`）。
`-DCALIB_WARNINGS_AS_ERRORS=ON` 可开启 `-Werror`，当前代码零警告通过。

## 模块

- `include/calibrator/calibration.hpp` / `src/calibration.cpp`
  - 损失：`L(X) = mean exp(-|x| / s)`，`s = rms(X)`（尺度不变）。
    该目标在激活被旋转到通道间均衡、远离 0 时下降，从而抑制极端/死通道。
  - 搜索：Jacobi/Givens 风格，对随机配对的通道对做一维角度搜索
    （`[0, π)` 48 点网格 + 6 次黄金分割局部细化），只接受严格改善的角度。
  - 每次接受后对旋转矩阵做两遍修正 Gram–Schmidt 重新正交化，并用
    `Xr = X * R` 精确重建工作矩阵，避免浮点漂移累积。
  - 返回“有效历史”中损失最小的旋转：初始 `R0` 始终是候选；迭代无改善时
    原样返回 `R0`；非有限候选永远不会替换当前最优。常量、秩亏、全零输入
    均保持有限（返回有限正交矩阵与有限损失），但不承诺任意数据都有改善。
- `include/calibrator/quantize.hpp` / `src/quantize.cpp`
  - 位宽 `b ∈ {2,4,8}`，整数范围固定 `±(2^(b-1)-1)`（即 ±1/±7/±127）。
  - 按行、连续列分组（`group_size` 由调用方选择），组尾不足整组照常处理。
  - 对称量化 `q = round(x / scale)`，`scale = max|x| / qmax`；
    全零分组 `scale = 0`、码全为 0，反量化严格还原为零。
  - 提供反量化矩阵与 `rmse / mean_abs / max_abs / relative_rmse` 误差统计。
- `include/calibrator/pack.hpp` / `src/pack.cpp`
  - 有符号码的 `bits` 宽小端位域打包（bit 0 为首个码的 LSB），每行填充到
    整字节；解包做有符号扩展，短尾组/奇数行宽往返一致；越界码与过短缓冲报错。
- `include/calibrator/io.hpp` / `src/io.cpp`
  - 校准参数（旋转、配置、损失、计数）保存为自描述小端二进制 `OQCB`，
    尾部 FNV-1a 64 位校验和；加载时校验 magic/版本/长度/维度/校验和/正交性，
    损坏、截断、外来文件都会明确抛错。

### 线性输出等价

`Y = X W = (X R)(R^T W)`。旋转激活 `X*R` 与同步变换权重 `R^T*W` 在未量化时
保持原线性输出；量化只发生在两者分别分组量化之后。

## 数值容限（测试中实际断言）

- 正交性：`max|R^T R - I| < 1e-5`（输入检查与结果均如此，单精度实现）。
- 未量化输出等价：相对 Frobenius 误差 `< 1e-4`（demo 实测约 `1e-7`）。
- 量化误差上界：任意元素 `|x - dq(x)| <= scale/2`（全零组为严格 0）。
- 工件往返：旋转矩阵最大差 `< 1e-7`，损失字段 `1e-12`。
- 打包往返：逐码精确相等（0 误差）。

注意：2/4 bit 量化不保证误差一定下降；测试只断言范围、对称性、全零/尾组
行为、位宽单调性（8bit 不劣于 4bit 不劣于 2bit）与有限统计，不虚构“必降”。

## 输入约束与错误

- 通道维度（激活列、旋转阶数、权重列）≤ 16；行数 ≤ 65536；矩阵非空。
- 初始旋转必须为与通道数匹配的方阵且正交（容差 `1e-5`）。
- 位宽非法、`group_size <= 0`、维度冲突、非有限（NaN/Inf）输入均抛出带明确
  英文信息的 `std::invalid_argument` / `std::runtime_error`。

## 测试

6 个可执行套件共 30 个 `TEST_CASE`，另加 1 个 CTest 元测试
（`meta_no_zero_tests`）：实际执行所有套件并解析注册用例数，任何套件失败、
注册 0 个用例、或总数少于 15 都判定失败——零测试/空壳测试无法通过。

```bash
ctest --test-dir build --output-on-failure
```
