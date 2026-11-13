# ForGeNet

ForGeNet is a two-stage framework for generalizable deepfake detection with limited AI-generated training images and abundant real images. Stage 1 learns representations through Leading Gradient Rebalancing (LGR), combining asymmetric contrastive objectives with an adaptive gate. Stage 2 learns the decision boundary with the lightweight Multi-Scale Forgery Enhancement (MSFE) module.

## Installation

```bash
git clone https://github.com/zuiluorenjian/ForGeNet.git
cd ForGeNet

conda create -n forgenet python=3.9 -y
conda activate forgenet
pip install -r requirements.txt
```

## Model Weights

Pretrained model weights are available on [Baidu Netdisk](https://pan.baidu.com/s/1r7FwyO5toccjB0FATXEh8A?pwd=6ua4) (extraction code: `6ua4`).

## Dataset

Organize the training data as follows:

```text
/path/to/datasets/your_dataset/
├── train/
│   ├── 0_real/
│   └── 1_fake/
└── val/
    ├── 0_real/
    └── 1_fake/
```

Use a prepared 95:5 real-to-AI-generated training split. Run the commands below from the repository root. Training hyperparameters are defined in the ProGAN shell scripts. Evaluation subsets are configured in `dataset_paths.py`; `--test_root_override` sets their common root directory.

## Stage 1: LGR Representation Learning

```bash
bash run_stage1_ProGAN.sh \
  --wang2020_data_path /path/to/datasets/your_dataset
```

The best Stage-1 checkpoint is saved as `./checkpoints/stage1/best_model.pth`.

## Stage 2: MSFE Decision Boundary Learning

```bash
bash run_stage2_ProGAN.sh \
  --pretrain_model_path ./checkpoints/stage1/best_model.pth \
  --wang2020_data_path /path/to/datasets/your_dataset \
  --test_root_override /path/to/datasets/Ojha
```

Checkpoints are saved to `./checkpoints/stage2_msfe/`, including `model_best_test.pth` and `model_final.pth`.

## Inference

The test directory can contain `0_real/` and `1_fake/` directly, or one subdirectory per generator containing those folders.

```bash
python evaluate_msfe.py \
  --checkpoint /path/to/ForGeNet.pth \
  --data-root /path/to/datasets/test \
  --batch-size 8 \
  --output ./results/evaluation.json
```

Results are saved to JSON and TSV files. Evaluation uses center crop 224 by default; add `--transform resize224` for OpenSDI.

## Monitoring

```bash
pip install tensorboard
tensorboard --logdir ./logs
```
