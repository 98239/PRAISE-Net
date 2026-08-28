# PRAISE-Net

English | [简体中文](README_zh-CN.md)

Official training-code package for **PRAISE-Net: Proposal-guided Relative
Anatomy and Intra-lesion Semantics Enhancement Network**, developed for
single-slice endometrial-cancer lesion segmentation on CT images.

This repository contains only the code required to train the manuscript's main
PRAISE-Net model. It does not contain comparison models, ablation variants,
data-splitting code, patient assignments, patient images, segmentation masks,
pretrained weights, or manuscript result files.

## Architecture

The public code uses the same names as the manuscript.

| Module | Full name | Role |
|---|---|---|
| CPG | Context-Preserving Lesion Proposal Generator | Preserves full-resolution local cues, expands the receptive field through three downsampling stages, aggregates same-scale multi-range context, and reconstructs one spatially aligned lesion proposal. |
| MTRG | Multi-scale Transition Reliability Guidance | Estimates four scale-specific reliability fields from local contrast and cross-neighborhood transitions, then modulates scale-aligned feature transfer during proposal reconstruction. |
| RAISE | Relative Anatomy and Intra-lesion Semantics Enhancer | Applies training-time boundary, signed relative-position, and relative-region supervision to the shared high-resolution representation. |

RAISE does not add a terminal correction head during inference. The final mask
is obtained only from the unified proposal logits followed by sigmoid and the
fixed threshold.

## Repository structure

| Path | Purpose |
|---|---|
| praise_net/model.py | PRAISE-Net architecture with CPG, MTRG, RAISE, and the unified proposal head. |
| praise_net/losses.py | Joint segmentation, boundary, signed-distance, and relative-region objectives used for the main model. |
| praise_net/data.py | Paired CT/mask loading, resizing, augmentation, patient-ID parsing, and patient-balanced sampling weights. |
| praise_net/metrics.py | Patient-volume aggregation followed by patient-macro Dice, IoU, Precision, Recall, Accuracy, and HD95. |
| praise_net/train.py | Main five-fold training, checkpoint selection, early stopping, final evaluation, and result summarization. |
| tests/test_smoke.py | Forward, loss, gradient, output, and parameter-count smoke test. |
| data/five_fold | Empty directory structure to be populated after external patient-level five-fold splitting. |
| outputs | Empty placeholder for checkpoints and metrics. |

## Dataset

PRAISE-Net was evaluated on the CT semantic-segmentation component of the
public **ECPC-IDS** dataset:

- Dataset record: [ECPC-IDS on Figshare](https://doi.org/10.6084/m9.figshare.23808258)
- Dataset paper: Tang D, Li C, Du T, et al. *ECPC-IDS: A benchmark
  endometrial cancer PET/CT image dataset for evaluation of semantic
  segmentation and detection of hypermetabolic regions*. Computers in Biology
  and Medicine. 2024;171:108217.
  [https://doi.org/10.1016/j.compbiomed.2024.108217](https://doi.org/10.1016/j.compbiomed.2024.108217)

The ECPC-IDS files are not redistributed in this repository. Download the
dataset from its official record and comply with the access and licence terms
shown there.

For this experiment, use the lesion-containing CT PNG images and their paired
semantic-segmentation masks. PET images, DICOM files, object-detection XML
annotations, and other modalities are not read by this training package.
Every CT image must have exactly one mask with the same filename stem. The
training loader treats the first three digits of the ECPC-IDS slice filename as
the patient identifier.

## Installation

Python 3.10 or later is recommended. Install a CUDA-enabled PyTorch build
appropriate for the local CUDA driver, then install this package.

Clone or download this repository from its GitHub page, then run:

~~~bash
cd PRAISE-Net
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

On Windows PowerShell, activate the environment with:

~~~powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
~~~

For development and smoke tests:

~~~bash
python -m pip install -e ".[dev]"
pytest -q
~~~

## Prepare the patient-level five-fold data

Download the official dataset and perform the five-fold partition outside this
repository. The split must use the patient, rather than the individual slice,
as the allocation unit. All slices belonging to one patient must remain in the
same validation fold so that no patient appears in both the training and
validation subsets of a fold.

After splitting, place the CT images and masks in the provided empty directory
structure:

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

Folders fold_2 through fold_5 must contain the same train/ct, train/label,
val/ct, and val/label subdirectories shown for fold_1. The code does not
generate or publish a patient-to-fold list. Users are responsible for verifying
that CT/mask stems match, that the five validation patient sets are mutually
exclusive, and that every patient is used for validation exactly once.

## Train PRAISE-Net

The complete five-fold experiment is launched with one command:

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 2 3 4 5 --epochs 300 --batch-size 16 --gradient-accumulation-steps 1 --image-size 256 --learning-rate 3e-4 --weight-decay 1e-4 --patience 30 --workers 4 --seed 42 --threshold 0.5 --device cuda --amp --deterministic
~~~

To train only one fold:

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 --device cuda --amp --deterministic
~~~

To skip folds for which metrics.json already exists:

~~~bash
python -m praise_net.train --data-root data/five_fold --output-root outputs --folds 1 2 3 4 5 --device cuda --amp --deterministic --skip-completed
~~~

Use --gradient-checkpointing if GPU memory is limited. Reducing batch size or
increasing --gradient-accumulation-steps changes the optimization trajectory
and should be documented.

## Training protocol

- Input: single CT slice, shape [B, 1, 256, 256].
- Image interpolation: bilinear; mask interpolation: nearest neighbor.
- Intensity range after loading: [0, 1].
- Augmentation: synchronized horizontal flip, intensity scaling/offset, and
  light Gaussian noise.
- Sampling: patient-balanced WeightedRandomSampler on the training set.
- Optimizer: AdamW, learning rate 3e-4, weight decay 1e-4.
- Scheduler: cosine annealing to 1e-6.
- Maximum epochs: 300; early-stopping patience: 30.
- Checkpoint criterion: maximum validation patient-macro Dice.
- Inference threshold: 0.5, without threshold search or connected-component
  post-processing.
- Fold summary: mean and sample standard deviation across the five folds.
- HD95 unit: voxel distance on the 256 × 256 resampled grid, not millimetres.

The joint loss is:

~~~text
L_total = 1.35 L_seg + 0.20 L_boundary
          + 0.10 L_distance + 0.15 L_relative-region
~~~

The evaluated model uses one shared set of proposal logits as its final output.
Consequently, the unit-weight final segmentation term and the 0.35-weight
proposal segmentation term are consolidated as 1.35 L_seg rather than treated
as losses from two independent prediction branches.

L_boundary jointly supervises the RAISE boundary output and the four MTRG
reliability fields. The signed-distance target uses radii {1, 2, 4, 8};
distances not covered by these neighborhoods are truncated at 16 pixels and
all distances are normalized by 16.

## Outputs

Each fold writes:

~~~text
outputs/praise_net/fold_<n>/
├── best.pth
├── history.csv
└── metrics.json
~~~

The checkpoint stores the model, optimizer and scheduler states, best epoch,
best patient-macro Dice, parameter count, module names, and complete training
configuration. The experiment also writes:

- outputs/praise_net/summary.csv: one row per fold;
- outputs/praise_net/five_fold_summary.json: five-fold means and sample standard
  deviations when folds 1–5 are run together.

## Reproducibility notes

- The public model has 6,514,682 trainable parameters.
- The code fixes Python, NumPy, and PyTorch random seeds and enables
  deterministic PyTorch/CuDNN behavior when --deterministic is used.
- Exact floating-point reproduction can still depend on the operating system,
  GPU, CUDA, cuDNN, and PyTorch version.
- The data and output directories are intentionally ignored by Git. Do not
  commit ECPC-IDS images, masks, or local checkpoints to this repository.

## Citation

If this code is useful, cite the PRAISE-Net article after its bibliographic
record becomes available and cite the ECPC-IDS dataset and dataset paper listed
above.

## License

The PRAISE-Net source code in this repository is released under the MIT
License. See [LICENSE](LICENSE) for the complete terms. The ECPC-IDS dataset is
not part of this software distribution and remains governed by the terms stated
on its official Figshare record.
