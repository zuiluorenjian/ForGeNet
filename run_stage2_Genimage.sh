#!/bin/bash


echo "==================================="
echo "   第二阶段：平衡数据集微调"
echo "   数据集：Genimage 95:5"
echo "==================================="

DATA_PATH="/opt/data/private/ysb/NPR-DeepfakeDetection/dataset/SDv1.4"

PRETRAIN_MODEL_PATH="/opt/data/private/ysb/project/CVPR/UniversalFakeDetect/experiments/SDv1.4_95:5_stage1_20250927_140335/checkpoints/best_model.pth"

EPOCHS=70
LR=1e-4
REAL_COUNT=810
FAKE_COUNT=810

BASE_DIR="./experiments/SDv1.4_95:5_stage2_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${BASE_DIR}/checkpoints"
LOG_PATH="${BASE_DIR}/logs"

echo "数据路径: $DATA_PATH"
echo "预训练模型: $PRETRAIN_MODEL_PATH"
echo "保存路径: $BASE_DIR"

if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
    echo "错误：预训练模型不存在: $PRETRAIN_MODEL_PATH"
    echo "请确保第一阶段训练已完成并生成了best_model.pth文件"
    exit 1
fi

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
echo "训练集图片数量: $(find $DATA_PATH/train -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)"
echo "验证集图片数量: $(find $DATA_PATH/val -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)"

echo "开始第二阶段微调训练..."

python train_stage2.py \
    --pretrain_model_path "$PRETRAIN_MODEL_PATH" \
    --wang2020_data_path "$DATA_PATH" \
    --data_mode "wang2020" \
    --epochs $EPOCHS \
    --lr $LR \
    --real_count $REAL_COUNT \
    --fake_count $FAKE_COUNT \
    --save_path "$SAVE_PATH" \
    --log_path "$LOG_PATH" \
    --patience 15 \
    --save_freq 5

if [ $? -eq 0 ]; then
    echo "==================================="
    echo "   第二阶段微调训练完成！"
    echo "==================================="
    echo "实验目录: $BASE_DIR"
    echo "模型保存在: $SAVE_PATH"
    echo "日志保存在: $LOG_PATH"
    echo ""
    echo "查看训练日志:"
    echo "  tensorboard --logdir $LOG_PATH"
    echo ""
    echo "模型文件:"
    echo "  最佳模型: $SAVE_PATH/model_best.pth"
    echo "  最终模型: $SAVE_PATH/model_final.pth"
    echo "==================================="
    echo ""
    echo "🎉 两阶段训练完成！可以开始测试模型性能了。"
else
    echo "❌ 第二阶段微调训练失败！请检查错误信息。"
    exit 1
fi

echo ""
echo "💡 下一步 - 测试模型:"
echo "可以使用训练好的模型进行测试和评估"
echo "测试脚本示例:"
echo "python test_model.py \\"
echo "    --model_path \"$SAVE_PATH/model_best.pth\" \\"
echo "    --test_data_path \"$DATA_PATH/test\" \\"
echo "    --arch \"CLIP:ViT-L/14\""
