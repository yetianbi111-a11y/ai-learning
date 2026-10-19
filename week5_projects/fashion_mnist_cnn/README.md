# projects/fashion_mnist_cnn/

data.py
model.py
train.py
evaluate.py
predict.py
README.md
requirements.txt
outputs/             # 本地结果；大权重不提交 Git


# Fashion-MNIST CNN

基于 PyTorch 的 Fashion-MNIST 图像分类项目。

## 环境安装

```bash
pip install -r requirements.txt
```

## 使用方式

### 训练

```bash
python train.py --epochs 10 --batch-size 64 --lr 1e-3
```

### 评估

```bash
python evaluate.py
```

### 推理

```bash
# 随机从测试集抽样预测
python predict.py

# 预测指定图片
python predict.py --image path/to/image.png
```

## 目录说明


| 文件          | 作用                   |
| ------------- | ---------------------- |
| `data.py`     | 数据集加载与预处理     |
| `model.py`    | CNN 模型定义           |
| `train.py`    | 训练循环、保存最佳权重 |
| `evaluate.py` | 测试集指标与混淆矩阵   |
| `predict.py`  | 单张图片推理           |
| `outputs/`    | 模型权重、日志、结果   |

## 模型结构

```
Conv(1→6, 5×5, pad=2) → ReLU → MaxPool(2)     # 28×28 → 14×14
Conv(6→16, 5×5)       → ReLU → MaxPool(2)     # 14×14 → 5×5
Flatten → Linear(400→120) → ReLU → Dropout
        → Linear(120→84)  → ReLU → Dropout
        → Linear(84→10)
```

## 预期结果

- 测试集准确率：**~90%**（10 epochs）
- 最佳模型保存在 `outputs/best_model.pth`
