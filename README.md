# ForGeNet

ForGeNet is a two-stage framework for universal AI-generated image detection. Stage 1 learns robust representations with supervised contrastive learning. Stage 2 performs parameter-efficient fine-tuning with the Multi-Scale Feature Enhancement (MSFE) module.

## Installation

```bash
git clone https://github.com/zuiluorenjian/ForGeNet.git
cd ForGeNet

conda create -n forgenet python=3.9 -y
conda activate forgenet
pip install torch torchvision numpy pillow opencv-python scipy scikit-image scikit-learn tensorboardX tqdm ftfy regex
```

## Model Weights

Pretrained model weights are available on [Google Drive](https://drive.google.com/drive/folders/15r3zb8bBaYwiSUvy_uRyXL1OpiqFeADA?usp=sharing).

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

Replace `/path/to/datasets` in `dataset_paths.py` with the location of your evaluation datasets when needed.

## Stage 1: Contrastive Pretraining

```bash
python train_stage1.py \
  --wang2020_data_path /path/to/datasets/your_dataset \
  --data_mode wang2020 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-3 \
  --arch CLIP:ViT-L/14 \
  --output_dim 128 \
  --hidden_dim 768 \
  --batch_norm \
  --temperature 0.07 \
  --base_temperature 0.07 \
  --save_path ./checkpoints/stage1 \
  --log_path ./logs/stage1
```

The best Stage 1 checkpoint is saved as `./checkpoints/stage1/best_model.pth`.

## Stage 2: MSFE Fine-Tuning

```bash
python train_stage2_msfe.py \
  --pretrain_model_path ./checkpoints/stage1/best_model.pth \
  --wang2020_data_path /path/to/datasets/your_dataset \
  --test_data_path /path/to/datasets/evaluation_datasets \
  --data_mode wang2020 \
  --epochs 2 \
  --batch_size 16 \
  --lr 2e-4 \
  --real_count 810 \
  --fake_count 810 \
  --msfe_hidden_dim 128 \
  --msfe_kernel_sizes 3 5 7 \
  --msfe_dropout 0.1 \
  --msfe_scale 0.1 \
  --num_adapted_layers 24 \
  --save_path ./checkpoints/stage2_msfe \
  --log_path ./logs/stage2_msfe
```

The final checkpoint is saved as `./checkpoints/stage2_msfe/model_final.pth`.

## Monitoring

```bash
tensorboard --logdir ./logs
```

## Notes

- Use `0_real` for real images and `1_fake` for generated images.
- Adjust batch sizes according to available GPU memory.
- CLIP weights are loaded automatically by the model implementation.
