

## 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 项目结构

```text
configs/
  task1.yaml
  task2.yaml
  mtl.yaml
docs/
  reproducibility.md
  pr_template_fill_example.md
src/
  datasets/
    __init__.py
    cifar.py
    denoising.py
  models/
    __init__.py
    common.py
    task1_denoiser.py
    task2_classifier.py
    mtl.py
  training/
    __init__.py
    task1.py
    task2.py
    mtl.py
    eval.py
  utils/
    __init__.py
    config.py
    io.py
    metrics.py
    seed.py
run_all.ps1
requirements.txt
README.md
```

## 3. 训练流程

先设置 Python 模块路径：

```bash
set PYTHONPATH=src
```

### 3.1 Task1 去噪

```bash
python -m training.task1 --config configs/task1.yaml
```

### 3.2 Task2 分类

```bash
python -m training.task2 --config configs/task2.yaml
```

### 3.3 多任务微调

```bash
python -m training.mtl --config configs/mtl.yaml
```

## 4. 评估

```bash
python -m training.eval --config configs/task1.yaml --mode task1
python -m training.eval --config configs/task2.yaml --mode task2
python -m training.eval --config configs/mtl.yaml --mode mtl
```

## 5. 网络结构详细介绍

本网络采用共享encoder加上分类头和去噪头的结构，其中task1是去噪任务，task2是分类任务，
其中shared_encoder采用6层残差块，每个block内部采用一层卷积，然后归一化以后（BN），然后激活（relu），然后再进行一次卷积和归一化的操作，并且6个block均进行残差学习
去噪头有4层block,每一层手写了一个残差学习，每层都是卷积，relu，卷积的结构，网络学习的是噪声的特征，最后手写输入减去噪声就是输出
分类头先进行全局自动池化，展平输出logits，依据类别分类
shared_features 复制成去噪/分类两路，再用 1x1 Conv 做双向残差加和：
f_d = f_d + P(c->d)(f_c)，f_c = f_c + P(d->c)(f_d)。分类分支先进行预测，然后用gate
调制去噪特征，guided = denoise_features * (1 + gate(logits))。
然后分类，去噪一起输出总损失 total_loss = ["lambda_denoise"] * denoise_loss+["lambda_cls"] * cls_loss+["lambda_consistency"] * consistency_loss
