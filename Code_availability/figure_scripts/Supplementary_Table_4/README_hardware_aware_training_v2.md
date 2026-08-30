# Hardware-aware training @ (0, 0) dBm — 串行 (serial) 编码 2×2 实验（写文章用 / N=600）

**来源** repo `QPG-MIT/MIWEN_Mingran`，分支 **`twin/high_power-rig`**（不在 main 上）
**归档自 commit** `2c938ef47862a336d145293c65553907c11b5d38`（2026-08-26，作者 jon-morag）
**实验日期** 2026-08-23 ~ 08-26
**仓库原始位置** 所有文件原本都在 `share_20260712/`（外加一份 `docs/plans/`）

> **与 v1 的区别**：v1 里把 images 0–299 和 images 300–599 当成「主实验 + 复现」两个结果分开报。
> **v2 只报一个结果：N=600**（两段合并）。代码、权重、模型、支撑分析全部照旧保留，力求完整。
>
> **命名说明**：文件名一律保持仓库原名，便于回溯 git 历史；分类与含义见下面的表格。

---

## 1. 结果（唯一报告口径：N=600）

2×2 实验：{clean, twin} × {digital, hardware}，同一套架构、同一套硬件协议。

- **clean** = 按理想乘法器训练（不知道器件非线性）
- **twin** = **hardware-aware**：训练时每一个乘积都先过一遍测量标定的 Shockley 型**数字孪生** `f(x,w)`，再求和
- 两条支路**都没有噪声注入、没有 bias correction**；**推理路径里不含任何 twin 成分**

工作点 **(LO 0 dBm, RF 0 dBm)**，链路为已装的 10 dB IF pads。

### 主表

| 支路 | digital | **hardware (N=600)** |
|---|---|---|
| clean（理想乘法器训练） | 99.50 %（ideal） | **5.67 %** (34/600) |
| **twin（hardware-aware）** | 98.50 %（under twin） | **98.50 %** (591/600) |

补充数字：

- clean 支路坍缩成常数分类器：590/600 全预测同一类（class 12）。
- clean-under-twin 的 digital 预报是 **0.83 %** (5/600) —— twin **提前预报**了 clean 网络会崩。
- twin 支路的 hardware **98.50 % 与它自己的 digital comparator 98.50 % 完全重合**（都是 591/600）。
- 逐层标定 relRMSE：twin `[0.142, 0.087, 0.021, 0.162]` vs clean-vs-ideal `[0.53, 0.61, 0.32, 2.94]`。

> 上面每个数字都能用根目录的 **`verify_accuracy.py`** 从原始预测文件重算（只依赖 numpy）：
> `python verify_accuracy.py`
> 脚本内含断言：两条支路跑的是**同一批 600 张图**、每条支路**只用了一个权重文件**、**只有一个功率点**、图片**无重复**。

### 这 600 张图是怎么来的

- 取自冻结的 1200 张 GTSRB battery 的前 600 张（`battery[0:600]`），两条支路完全同一批图。
- 分成两个**互不重叠**的 300 张段跑：images 0–299（8/23–24）与 images 300–599（8/25–26）。
- **两段用的是同一组权重、同一套冻结标定**（log 里是 `reloaded frozen calibration`，不是重新标定）。
- 第二段的 digital comparator 在上硬件**之前**就已提交（预注册）。
- 每段每条支路约 **8.5 小时**机时（log 计时 ~30,400 s），所以停在 600 而不是做满 1200。

---

## 2. 文件夹结构

```
hardware_aware_training_v2/
├── README.md                                     本文件
├── verify_accuracy.py                            从原始预测重算 N=600 结果（本次整理新增）
├── 00_report_and_audit/                          审计报告与预注册协议
├── 01_digital_twin_model/                        数字孪生模型（训练用的物理模型）
├── 02_training_code/                             训练代码
├── 03_training_results/                          训练结果（日志 + 权重）
│   ├── weights_twin_15ep_FIELDED/                ← 真正上机的 twin 权重
│   ├── weights_twin_60ep_sibling/                ← 60-epoch 对照（未上机）
│   └── weights_clean_baseline/                   ← clean 支路权重
├── 04_hardware_test_code/                        硬件测试代码
├── 05_hardware_result_0dBm_N600/                 硬件测试结果（唯一报告口径）
│   ├── clean_arm/
│   └── twin_arm_hw_aware/
├── 06_digital_comparators/                       上机前钉死的 digital 预测
├── 07_supporting_analysis_enob_and_driveladder/  支撑分析：ENOB + drive ladder
└── 08_frozen_inputs_and_labels/                  冻结输入：评测索引与标签
```

---

## 3. 各文件说明

### 00_report_and_audit — 审计报告与预注册协议

| 文件 | 说明 |
|---|---|
| `serial_run_audit.pdf` / `.tex` | **主报告**（v3.2）：claim、协议、twin 物理、全部方程、与 comb 认证 run 的逐条差异、gates、结果表 |
| `2026-08-24_serial_m10m24_prespec.md` | 上硬件前冻结的 session pre-spec（含第二段的复现计划与逐条 RESULT 记录） |

### 01_digital_twin_model — 数字孪生模型

| 文件 | 说明 |
|---|---|
| `serial_twin_fit.py` | twin 拟合代码：product core（两个 one-pole knee）× K=20 tanh ridges，拟合在端口标定过的 USRP CW map 上 |
| `serial_twin_model.json` | **冻结的 twin 曲面**，sha256[:16] = `b07bb82b328d9c8e`（与 frozen reference 内的 pin 一致）。held-out 0.12 dB；serial-slot 交叉验证最差 bin 3.7 % |
| `heatmap_unpadded_20260814.npz` | 拟合的输入数据：USRP CW map |
| `twin_ksweep.py` / `twin_ksweep_20260825.json` / `.png` | ridge 数 K 的扫描（论证 K≈20–30 之后不再有收益） |

### 02_training_code — 训练代码

| 文件 | 说明 |
|---|---|
| `train_serial_twin_short.py` | **实际上机使用的训练脚本**（15-epoch sprint）。twin 在训练中作用于求和前的每一个乘积；同时钉死 clean-under-twin 预测 |
| `train_serial_twin.py` | 60-epoch 对照（同一 recipe，更长训练） |
| `ladder_cnn_v2.py` | 模型与训练库：`ARCHS["r3plus"]`、训练 recipe、`export_runner`（BN folding） |
| `miwen_serial_frozen_reference.py` | **冻结算法单文件**：预处理、权重加载、ideal/twin 前向、twin 曲面、slot 编码、chirp sync、解码、判决。`--verify` 自检 sha 与 bit-exact 回放 |

训练脚本读取：`serial_twin_model.json`、`r35_r3plus_s0_hw.npz`（取架构）、`battery_random1200_idx.npy`，以及数据集缓存 `gtsrb_roi_32x32.npz`（**该缓存未在仓库中**，见第 4 节）。

### 03_training_results — 训练结果

| 文件 | 说明 |
|---|---|
| `train_serial_twin_short.log` | **上机权重的训练日志**：ep5 99.36 → ep15 99.87 (best)；SELECTED ckpt: test full **98.56**、test N=300 **98.33** |
| `train_serial_twin.log` | 60-epoch 日志：best val 99.92；SELECTED ckpt: test full **99.07**、N=300 **99.67** |
| `weights_twin_15ep_FIELDED/serial_twin_s0_hw.npz` | **真正上硬件的 twin 权重**，sha256[:16] = `825ab4ee148dc1cd` |
| `weights_twin_15ep_FIELDED/gtsrb_r3plus_serialtwin_short_s0.npz` | 对应的训练 checkpoint（导出前） |
| `weights_twin_60ep_sibling/serial_twin_s0_ep60_hw.npz` | 60-epoch 版权重（**从未上机**，作为训练量对照保留） |
| `weights_twin_60ep_sibling/gtsrb_r3plus_serialtwin_s0.npz` | 对应训练 checkpoint |
| `weights_clean_baseline/r35_r3plus_s0_hw.npz` | **clean 支路权重**，sha256[:16] = `8a4c9efddbf8cfd0`，即 comb 认证期的 checkpoint |

> twin 只训了 15 epoch，而 clean 支路是 150 epoch，这个训练量不对称在审计报告中已明确 disclose；60-epoch sibling 就是为此保留的对照。

### 04_hardware_test_code — 硬件测试代码

| 文件 | 说明 |
|---|---|
| `serial_nn_runner.py` | **主测试驱动 (v3)**：USRP 收发、采集、分块、gates、增量保存。`POWER = (0.0, 0.0)` 即 (0,0) dBm；`ARM_W` 把 clean→`r35_r3plus_s0_hw.npz`、twin→`serial_twin_s0_hw.npz`。用法：`python serial_nn_runner.py {clean|twin} <img_start> <img_end>` |
| `serial_nn_runner_ep60.py` | 同上，只差 3 行（指向 60-epoch 权重与对应输出/标定文件名）。**未用于本结果** |
| `miwen_serial_frozen_reference.py` | 每一个算法步骤都取自此文件（与 02 中为同一份） |
| `miwen_frozen_reference.py` | comb 期的冻结参考（runner 以 `ref1` 导入） |
| `sync_v2.py` | chirp 同步 |
| `check_rig_free.py` | 上机前的设备占用检查 |

### 05_hardware_result_0dBm_N600 — 硬件测试结果（唯一报告口径）

**clean_arm/** — 合计 600 张，5 个预测文件（runner 增量保存，互不重叠）

| 文件 | 覆盖 | 说明 |
|---|---|---|
| `serial_nn_clean_0_150_20260823.npz` | images 0–149 | 15 chunks |
| `serial_nn_clean_150_215_20260823.npz` | images 150–214 | 7 chunks |
| `serial_nn_clean_215_300_20260823.npz` | images 215–274 | 6 chunks |
| `serial_nn_clean_275_300_20260823.npz` | images 275–299 | 3 chunks |
| `serial_nn_clean_300_600_20260823.npz` | images 300–599 | 30 chunks |
| `serial_cal_clean_20260823.npz` | — | 冻结逐层标定（scales `s_l1..4` + gains `g_l1..4`），两段共用 |
| `serial_clean_h1c_20260824.log` | images 0–149 | session log |
| `serial_clean_h2a/h2b/h2c_20260824.log` | images 150–299 | session log |
| `serial_clean_00_fresh_20260826.log` | images 300–599 | session log（末行 `FINAL: 5.00`） |

**twin_arm_hw_aware/** — 合计 600 张，2 个预测文件

| 文件 | 覆盖 | 说明 |
|---|---|---|
| `serial_nn_twin_0_300_20260823.npz` | images 0–299 | 30 chunks |
| `serial_nn_twin_300_600_20260823.npz` | images 300–599 | 30 chunks |
| `serial_cal_twin_20260823.npz` | — | twin 支路冻结标定，两段共用 |
| `serial_twin_20260824.log` | images 0–299 | session log（末行 `FINAL: 97.67`） |
| `serial_twin_00_fresh_20260825.log` | images 300–599 | session log（末行 `FINAL: 99.33`） |

每个 `.npz` 内含 `chunk*_preds`、`sel`（评测图在 GTSRB test 中的全局 index）、`meta_json`（arm / 权重文件 / power / tslot / img_range / chain / readout）。

> log 里的 `FINAL:` 是**该段**的数字（97.67 / 99.33 / 5.00 等）。**报告口径是两段合并后的 N=600**，见第 1 节。

### 06_digital_comparators — 上机前钉死的 digital 预测

| 文件 | 说明 |
|---|---|
| `serial_predictions.json` | images 0–299 段的 pin：clean-under-twin 0.67 %、twin-under-twin 98.56 (full) / **98.33 (N300)**、twin val 99.87 |
| `digital_pins_seg300600.npz` | images 300–599 段的 pin（含逐图预测）：`clean_ideal` 100.00 / `twin_under_twin` 98.67 / `clean_under_twin` 1.00 |

### 07_supporting_analysis_enob_and_driveladder — 支撑分析

| 文件 | 说明 |
|---|---|
| `serial_enob.py` | 逐乘积 ENOB 分析（naive vs twin-inverted） |
| `serial_enob_00_20260824.json` | (0,0) 处结果：**naive 2.53 bits vs twin-inverted 4.38 bits**，含 8 个幅度 bin 的分解 |
| `serial_enob_00_20260824.png` | ENOB vs 乘积幅度 |
| `serial_enob_scatter_20260824.png` | 输出乘积 vs 输入乘积散点（twin 逆变换把 sigmoid 拉直到 identity） |
| `serial_stationA.py` + `serial_stationA_20260823.npz` | Stage A drive ladder：−9…+7 dBm 上逐乘积线性残差 0.21→0.67，并保存 ENOB 用的原始 payload |

### 08_frozen_inputs_and_labels — 冻结输入

| 文件 | 说明 |
|---|---|
| `battery_random1200_idx.npy` | 冻结的 1200 张评测图 index（全 campaign 共用）。本结果用前 600 张 |
| `battery_frozen_slim.npz` | 含 `img_index` → `labels` 的**真值标签**（`verify_accuracy.py` 用它算准确率） |
| `s4_random450_idx.npy` | 标定用图 index（runner 取前 N_CAL 张） |

---

## 4. 未随仓库归档的文件（原仓库即不含）

以下是代码会引用、但仓库里没有的大文件，需要时要去实验机上取：

- `gtsrb_roi_32x32.npz` — 完整 GTSRB 32×32 ROI 缓存（训练脚本与 runner 都读它）
- `serial_diag_cap.npy` — 原始采集回放（`serial_enob.py` 与 frozen reference 的 `--verify` 回放要用）

---

## 5. 一句话结论（摘自审计报告）

两项 twin 验证都通过：(i) clean 硬件坍缩，落在钉死的 clean-under-twin 预报上——twin **提前预报**了未经训练适配的网络的命运；(ii) twin 硬件落在 twin-under-twin 的 digital comparator 上（N=600 时两者都是 591/600）——**把 twin 放进训练回路，就是串行编码下的 hardware-awareness 机制**。结合 comb 程序的结果，说明*决定训练需要何种抽象的是编码方式，而不是器件本身*：非线性在求和之外只是读出细节，非线性在求和之内则要求训练回路里有一个物理 twin。
