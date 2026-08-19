#!/bin/bash


echo "==================================="
echo "   第一阶段：对比学习预训练"
echo "   数据集：SDv1.5 95:5"
echo "==================================="

DATA_PATH="/opt/data/private/ysb/NPR-DeepfakeDetection/dataset/SDv1.4"

EPOCHS=100
BATCH_SIZE=128
LR=1e-3
ARCH="CLIP:ViT-L/14"
OUTPUT_DIM=128
HIDDEN_DIM=768
TEMPERATURE=0.07

BASE_DIR="./experiments/SDv1.4_95:5_stage1_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${BASE_DIR}/checkpoints"
LOG_PATH="${BASE_DIR}/logs"

echo "数据路径: $DATA_PATH"
echo "保存路径: $BASE_DIR"

if [ ! -d "$DATA_PATH" ]; then
    echo "错误：数据路径不存在: $DATA_PATH"
    echo "请检查数据集是否已正确准备"
    exit 1
fi

if [ ! -d "$DATA_PATH/train" ]; then
    echo "错误：训练数据目录不存在: $DATA_PATH/train"
    echo "请确保数据集包含train子目录"
    exit 1
fi

if [ ! -d "$DATA_PATH/val" ]; then
    echo "错误：验证数据目录不存在: $DATA_PATH/val"
    echo "请确保数据集包含val子目录"
    exit 1
fi

mkdir -p "$SAVE_PATH"
mkdir -p "$LOG_PATH"

echo "数据集统计："
echo "训练集图片数量: $(find $DATA_PATH/train -name "*.jpg" -o -name "*.png" -o -name "*.JPEG" | wc -l)"
echo "验证集图片数量: $(find $DATA_PATH/val -name "*.jpg" -o -name "*.png" -o -name "*.JPEG" | wc -l)"

echo "开始训练..."

python train_stage1.py \
    --wang2020_data_path "$DATA_PATH" \
    --data_mode "wang2020" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --arch "$ARCH" \
    --output_dim $OUTPUT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --batch_norm \
    --temperature $TEMPERATURE \
    --base_temperature $TEMPERATURE \
    --min_class 1 \
    --ratio_supervised_majority 0.0 \
    --save_path "$SAVE_PATH" \
    --log_path "$LOG_PATH" \
    --patience 15 \
    --save_freq 10

if [ $? -eq 0 ]; then
    echo "==================================="
    echo "   第一阶段训练完成！"
    echo "==================================="
    echo "实验目录: $BASE_DIR"
    echo "模型保存在: $SAVE_PATH"
    echo "日志保存在: $LOG_PATH"
    echo ""
    echo "查看训练日志:"
    echo "  tensorboard --logdir $LOG_PATH"
    echo ""
    echo "下一步 - 运行第二阶段训练:"
    echo "  python train_stage2.py \\"
    echo "    --pretrain_model_path \"$SAVE_PATH/best_model.pth\" \\"
    echo "    --wang2020_data_path \"$DATA_PATH\" \\"
    echo "    --subset_ratio 0.5"
    echo "==================================="
else
    echo "❌ 第一阶段训练失败！请检查错误信息。"
    exit 1
fi

echo ""
echo "💡 提示：如果上面的训练不稳定，可以尝试平衡数据集策略："
echo "python train_stage1.py \\"
echo "    --wang2020_data_path \"$DATA_PATH\" \\"
echo "    --data_mode \"wang2020\" \\"
echo "    --use_ratio_sampling \\"
echo "    --real_ratio 0.5 \\"
echo "    --fake_ratio 0.5 \\"
echo "    --epochs $EPOCHS \\"
echo "    --batch_size $BATCH_SIZE \\"
echo "    --ratio_supervised_majority 1.0 \\"
echo "    --save_path \"./experiments/progan95_5_balanced/checkpoints\" \\"
echo "    --log_path \"./experiments/progan95_5_balanced/logs\""
