<div align="center">

# BRMSNet

### Remote Sensing Salient Object Detection

**多尺度特征融合 · 门控跳跃连接 · 全分辨率细节细化**

![Task](https://img.shields.io/badge/Task-Salient_Object_Detection-2563eb?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Remote_Sensing-0f766e?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Encoder](https://img.shields.io/badge/Encoder-PVTv2--B1-7c3aed?style=flat-square)

[项目概览](#overview) &nbsp;·&nbsp; [模型结构](#method) &nbsp;·&nbsp; [计算量](#results) &nbsp;·&nbsp; [快速开始](#getting-started) &nbsp;·&nbsp; [代码导航](#code-guide)

</div>

---

<a id="overview"></a>
## 项目概览

**BRMSNet** 面向遥感图像中的显著目标检测，从输入图像生成逐像素显著性预测。项目包含模型、QAMWS 多尺度监督、数据读取、训练、验证、测试及结果导出。

模型由 **PVTv2-B1 编码器、HSSD 解码器、CGAG 跳跃连接与 FDRM 全分辨率细化模块**组成。训练阶段使用四个预测头，推理阶段只执行主预测头。

<table align="center">
  <tr>
    <th align="center">13.8219M</th>
    <th align="center">13.2354G</th>
    <th align="center">512 × 512</th>
  </tr>
  <tr>
    <td align="center">参数量</td>
    <td align="center">MACs</td>
    <td align="center">默认输入尺寸</td>
  </tr>
</table>

<p align="center"><sub>当前实现的本地 CPU 分析结果，使用 timm 1.0.30 和 THOP；检测精度需要完成真实数据集训练后评估。</sub></p>

<a id="method"></a>
## 模型结构

<p align="center">
  <a href="assets/figures/framework.pdf">
    <img src="assets/figures/framework.png" width="100%" alt="BRMSNet 整体架构及 HSSD、MRIR、FDRM、CGAG 模块详细结构。">
  </a>
</p>
<p align="center"><em>BRMSNet 整体架构与 HSSD、MRIR、FDRM、CGAG 模块。点击图片可查看原始 PDF。</em></p>

| 模块 | 作用 | 实现 |
| :--- | :--- | :--- |
| **PVTv2-B1** | 提取四级多尺度特征，经通道适配进入解码器 | [BRMSNet](models/pvt_mkunet.py) |
| **HSSD** | 通过通道/空间注意力、多核倒残差模块和逐级上采样恢复空间分辨率 | [解码器](models/pvt_mkunet.py) |
| **CGAG** | 对编码器跳跃特征进行门控后，与解码器特征相加 | [GroupedAttentionGate](models/pvt_mkunet.py) |
| **FDRM** | 融合全分辨率浅层特征与解码特征，细化预测细节 | [FullResolutionRefinement](models/pvt_mkunet.py) |
| **QAMWS** | 根据预测质量分配四头损失权重，并施加质量门控混合指导 | [qamws.py](models/qamws.py) |

<details>
<summary><b>展开：多尺度监督与预测接口</b></summary>

`model(images, return_all=True)` 返回 `[main, eighth, quarter, half]`，四个 logits 均对齐到输入尺寸。默认 `model(images)` 只返回 `[main]`，不执行辅助预测头。

QAMWS 没有可学习参数。每张图先计算四个分割损失，再由停止梯度的质量分数分配权重。混合目标在概率空间计算，只有其分割损失低于接收头时才施加 KL 指导。前 10 轮使用均匀权重，不施加混合指导。

分割损失由结构损失、边界 Dice 和 Focal Tversky 组合，计算时排除补边区域。

</details>

<a id="results"></a>
## 计算量与验证状态

| 项目 | 当前数值或状态 |
| :--- | :--- |
| 参数量 | **13.8219M** |
| MACs | **13.2354G** |
| FLOPs | **26.4709G**，按 `1 MAC = 2 FLOPs` 计算 |
| 分析输入 | `1 × 3 × 512 × 512`，主头推理 |
| CPU 验证 | 已运行计算量分析，训练/评估入口可正常显示帮助 |
| 完整数据集训练与精度 | 尚未验证 |
| CUDA AMP | 需要 GPU 环境验证 |

运行以下命令重新测量；脚本使用随机初始化权重，不需要数据集或下载预训练权重：

```bash
python profile_cpu.py
```

当前仓库提供模型流程图。真实图像的预测可视化由评估脚本导出，输出目录为 `predictions_rsod/`。

<a id="getting-started"></a>
## 快速开始

### 1. 安装

克隆仓库，使用 Python 3.9 或更新版本创建环境：

```bash
git clone https://github.com/lumenworks0403/BRMSNet.git
cd BRMSNet
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

先按设备安装匹配的 PyTorch 与 torchvision，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

默认训练通过 timm 下载编码器预训练权重，首次运行需要联网。使用 `--no-pretrained` 可从随机初始化开始训练。

### 2. 准备数据

数据集需自行准备。以 `ORSSD` 为例，默认目录如下：

```text
data/rsod/ORSSD/
├── train/
│   ├── images/
│   └── masks/
├── val/                    # 可选
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

图像和掩码按文件主名配对。数据存放在其他位置时，通过 `--data-root` 指定 `ORSSD/` 的父目录。

<details>
<summary><b>展开：训练/验证划分与输入处理</b></summary>

没有 `val` 时，从 `train` 固定留出 10%，通过样本列表划分，不移动文件。已有 `val` 时直接使用，并检查训练和验证是否重叠。测试集不参与划分和检查点选择。

图像保持宽高比缩放并补边到 512×512。训练阶段旋转、水平翻转和垂直翻转的概率均为 0.5。数据划分使用独立的 `--split-seed`，各次运行共用同一划分。

</details>

### 3. 训练

```bash
python train_rsod.py --dataset ORSSD --device cuda
```

首次尝试可只运行一个种子：

```bash
python train_rsod.py --dataset ORSSD --device cuda --runs 1
```

| 默认设置 | 数值 |
| :--- | :--- |
| 训练轮数 / batch size | `200` / `8` |
| 编码器 / 新层初始学习率 | `3e-5` / `3e-4` |
| 优化器 / 学习率调度 | AdamW / 余弦衰减 |
| 随机种子 | `42`、`43`、`44`，默认运行三次 |
| QAMWS | `tau=0.2`、`eta=0.2`、`lambda_ms=0.6`、`lambda_mix=0.1` |

每次运行在 `model_pth/RUN_ID/` 保存 `split.json`、`qamws_weights.csv`、最后一轮与最佳检查点。

### 4. 评估

将 `RUN_ID` 替换为训练打印的运行标识，也就是 `model_pth/` 下对应的运行目录名：

```bash
python evaluate_rsod.py --dataset ORSSD --run-id RUN_ID --device cuda
```

评估验证集：

```bash
python evaluate_rsod.py --dataset ORSSD --run-id RUN_ID --split val --device cuda
```

程序自动读取检查点目录下的 `split.json`，也可用 `--split-file` 指定划分文件。评估输出包含预测图和 Excel 报告。

<details>
<summary><b>展开：检查点选择与评估设置</b></summary>

每轮验证使用 `0.4 * S_measure + 0.3 * Dice + 0.3 * Boundary_F1` 选择检查点。

验证与测试共用评估代码，在 512×512 输入网格上剔除补边后逐图像计算指标，使用原始 sigmoid 概率，不做逐图像 min-max 归一化。二值阈值固定为 0.5，边界匹配容差为两个像素。

预测图恢复到原图尺寸；报告指标仍使用输入网格。数据集、训练权重、预测图和报告已由 `.gitignore` 排除，不随代码提交。

</details>

<details>
<summary><b>展开：旧权重兼容性</b></summary>

旧检查点可通过 `train_rsod.py --warm-start PATH` 初始化兼容层。当前通道注意力统一使用压缩比 16，旧版最后两级注意力尺寸不同的参数会重新初始化。旧权重需要重新训练，不能直接作为当前版本的完整推理检查点。

旧类名 `PVTMKUNetB1` 保留为 `BRMSNet` 的兼容别名。

</details>

<a id="code-guide"></a>
## 代码导航

<details>
<summary><b>展开：仓库目录</b></summary>

```text
BRMSNet/
├── assets/
│   └── figures/
│       ├── framework.pdf       # 架构图原始 PDF
│       └── framework.png       # README 展示图
├── models/
│   ├── pvt_mkunet.py           # 模型、解码器、门控与细化
│   └── qamws.py                # 多尺度监督
├── utils/
│   ├── dataloader_rsod.py      # 数据读取、变换与划分
│   ├── losses.py               # 分割损失
│   ├── rsod_metrics.py         # 显著性评估指标
│   ├── saliency.py             # 补边处理、Dice 与边界指标
│   └── training.py             # 训练辅助与计算量分析
├── train_rsod.py               # 训练入口
├── evaluate_rsod.py            # 评估入口
├── profile_cpu.py             # CPU 计算量分析
├── requirements.txt
└── .gitignore
```

</details>

| 功能 | 文件 |
| :--- | :--- |
| 主模型 | [pvt_mkunet.py](models/pvt_mkunet.py) |
| 训练监督 | [qamws.py](models/qamws.py) · [losses.py](utils/losses.py) |
| 数据读取 | [dataloader_rsod.py](utils/dataloader_rsod.py) |
| 训练与评估 | [train_rsod.py](train_rsod.py) · [evaluate_rsod.py](evaluate_rsod.py) |
| 评估指标 | [rsod_metrics.py](utils/rsod_metrics.py) · [saliency.py](utils/saliency.py) |
| 计算量分析 | [profile_cpu.py](profile_cpu.py) |

查看可用参数：

```bash
python train_rsod.py --help
python evaluate_rsod.py --help
```

当前目录未附带自动化测试套件。此前曾在 CPU 环境使用合成数据验证模型前向、损失、梯度、训练、检查点保存及报告导出；完整数据集训练仍需独立验证。

---

<p align="center"><a href="#brmsnet">返回顶部 ↑</a></p>
