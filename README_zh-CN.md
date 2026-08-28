# PRAISE-Net

[English](README.md) | 简体中文

本仓库提供 **PRAISE-Net：Proposal-guided Relative Anatomy and Intra-lesion
Semantics Enhancement Network（提议引导的相对解剖与病灶内语义增强网络）**
的正式训练代码，用于单层CT图像中的子宫内膜癌病灶分割。

本仓库仅包含训练论文主模型所需的代码，不包含对比模型、消融变体、
数据划分脚本、患者划分名单、患者影像、分割掩膜、预训练权重或论文结果文件。

## 模型架构

公开代码中的模块名称与论文保持一致。

| 模块 | 英文全称 | 作用 |
|---|---|---|
| CPG | Context-Preserving Lesion Proposal Generator | 保留原始分辨率的局部线索，通过三次下采样扩大感受范围，在固定深层尺度聚合多范围上下文，并重建一个空间对齐的病灶提议。 |
| MTRG | Multi-scale Transition Reliability Guidance | 根据局部对比和跨邻域转变估计四个尺度的可靠性场，并在提议重建过程中调节尺度对齐的特征传递。 |
| RAISE | Relative Anatomy and Intra-lesion Semantics Enhancer | 在共享的高分辨率表征上施加训练期边界、符号相对位置和相对区域监督。 |

RAISE在推理阶段不增加末端校正头。最终掩膜仅由统一提议logits经过
sigmoid和固定阈值得到。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `praise_net/model.py` | PRAISE-Net模型结构，包括CPG、MTRG、RAISE和统一提议头。 |
| `praise_net/losses.py` | 主模型使用的分割、边界、符号距离和相对区域联合目标函数。 |
| `praise_net/data.py` | 配对CT/掩膜读取、尺寸调整、数据增强、患者编号解析和患者平衡采样权重。 |
| `praise_net/metrics.py` | 将切片聚合为患者体数据，并计算患者宏平均Dice、IoU、Precision、Recall、Accuracy和HD95。 |
| `praise_net/train.py` | 主模型五折训练、检查点选择、早停、最终评估和结果汇总。 |
| `tests/test_smoke.py` | 前向计算、损失、梯度、输出尺寸和参数量冒烟测试。 |
| `data/five_fold` | 空数据目录，需在仓库外完成患者级五折划分后填入数据。 |
| `outputs` | 用于保存检查点和评估结果的空目录。 |

## 数据集

PRAISE-Net在公开 **ECPC-IDS** 数据集的CT语义分割部分进行评估：

- 数据集页面：[Figshare上的ECPC-IDS](https://doi.org/10.6084/m9.figshare.23808258)
- 数据集论文：Tang D, Li C, Du T, et al. *ECPC-IDS: A benchmark
  endometrial cancer PET/CT image dataset for evaluation of semantic
  segmentation and detection of hypermetabolic regions*. Computers in Biology
  and Medicine. 2024;171:108217.
  [https://doi.org/10.1016/j.compbiomed.2024.108217](https://doi.org/10.1016/j.compbiomed.2024.108217)

本仓库不再分发ECPC-IDS数据。请从官方页面下载数据，并遵守该页面规定的
访问条件和许可条款。

本实验使用包含病灶的CT PNG图像及其配对语义分割掩膜。训练代码不会读取
PET图像、DICOM文件、目标检测XML标注或其他模态。每张CT图像必须有且仅有
一个同名掩膜。数据加载器将ECPC-IDS切片文件名的前三位数字作为患者编号。

## 安装

建议使用Python 3.10或更高版本。请先安装与本机CUDA驱动兼容的CUDA版
PyTorch，再安装本代码包。

从GitHub页面克隆或下载仓库后执行：

~~~bash
cd PRAISE-Net
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

Windows PowerShell环境使用以下命令激活虚拟环境：

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
~~~

如需运行开发测试：

~~~bash
python -m pip install -e ".[dev]"
pytest -q
~~~

## 准备患者级五折数据

请下载官方数据集，并在本仓库之外完成五折划分。划分单位必须是患者，不能是
单张切片。同一患者的所有切片必须位于同一个验证折中，确保每一折的训练集与
验证集不存在患者重叠。

完成划分后，将CT图像和掩膜放入仓库提供的空目录：

~~~text
data/five_fold/
├── fold_1/
│   ├── train/ct
│   ├── train/label
│   ├── val/ct
│   └── val/label
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
~~~

`fold_2`至`fold_5`均应包含与`fold_1`相同的`train/ct`、`train/label`、
`val/ct`和`val/label`子目录。代码不会生成或公开患者—折次对应名单。
使用者应自行确认CT与掩膜文件主名一致、五个验证患者集合互不重叠，并且每名
患者恰好被用于一次验证。

## 训练PRAISE-Net

使用以下单条命令启动完整五折实验：

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 2 3 4 5 --epochs 300 --batch-size 16 --gradient-accumulation-steps 1 --image-size 256 --learning-rate 3e-4 --weight-decay 1e-4 --patience 30 --workers 4 --seed 42 --threshold 0.5 --device cuda --amp --deterministic
~~~

仅训练一个折次：

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 --device cuda --amp --deterministic
~~~

跳过已经存在`metrics.json`的折次：

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 2 3 4 5 --device cuda --amp --deterministic --skip-completed
~~~

如果GPU显存有限，可添加`--gradient-checkpointing`。减小批量大小或增大
`--gradient-accumulation-steps`会改变优化轨迹，应在实验记录中说明。

## 训练方案

- 输入：单张CT切片，尺寸为`[B, 1, 256, 256]`。
- 插值：图像使用双线性插值，掩膜使用最近邻插值。
- 读取后的强度范围：`[0, 1]`。
- 数据增强：图像与掩膜同步水平翻转，以及图像强度缩放/偏移和轻度高斯噪声。
- 采样：在训练集上使用患者平衡的`WeightedRandomSampler`。
- 优化器：AdamW，学习率`3e-4`，权重衰减`1e-4`。
- 学习率调度：余弦退火至`1e-6`。
- 最大训练轮数：300；早停耐心值：30。
- 检查点选择标准：验证集患者宏平均Dice最大值。
- 推理阈值：0.5；不进行阈值搜索或连通域后处理。
- 五折汇总：报告五折均值和样本标准差。
- HD95单位：256×256重采样网格上的体素距离，不是毫米。

联合损失为：

~~~text
L_total = 1.35 L_seg + 0.20 L_boundary
          + 0.10 L_distance + 0.15 L_relative-region
~~~

被评估模型使用同一组提议logits作为最终输出。因此，权重为1的最终分割项与
权重为0.35的提议分割项合并表示为`1.35 L_seg`，而不是将其解释为两个独立
预测分支的损失。

`L_boundary`同时监督RAISE边界输出和四个MTRG可靠性场。符号距离目标采用
半径集合`{1, 2, 4, 8}`；未被这些邻域覆盖的位置截断为16像素，所有距离再
除以16进行归一化。

## 输出文件

每个折次生成：

~~~text
outputs/praise_net/fold_<n>/
├── best.pth
├── history.csv
└── metrics.json
~~~

检查点保存模型、优化器和调度器状态，以及最佳训练轮次、最佳患者宏平均Dice、
参数量、模块名称和完整训练配置。实验还会生成：

- `outputs/praise_net/summary.csv`：每个折次一行；
- `outputs/praise_net/five_fold_summary.json`：同时完成第1至第5折时，保存
  五折均值和样本标准差。

## 可复现性说明

- 公开模型包含6,514,682个可训练参数。
- 使用`--deterministic`时，代码会固定Python、NumPy和PyTorch随机种子，
  并启用PyTorch/CuDNN确定性行为。
- 操作系统、GPU、CUDA、cuDNN和PyTorch版本仍可能影响浮点数层面的完全复现。
- 数据和输出目录已被Git忽略。请勿向本仓库提交ECPC-IDS图像、掩膜或本地检查点。

## 引用

如果本代码对您的研究有帮助，请在PRAISE-Net论文书目信息正式发布后引用该
论文，并同时引用上文列出的ECPC-IDS数据集及其数据集论文。

## 许可证

本仓库中的PRAISE-Net源代码采用MIT License发布，完整条款见
[LICENSE](LICENSE)。ECPC-IDS数据集不属于本软件发布内容，仍受其Figshare
官方页面所列条款约束。
