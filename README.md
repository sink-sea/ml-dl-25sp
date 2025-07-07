# ML&DL HW: CNN + ViT
天津大学《机器学习与深度学习》课程2024-2025第二学期大作业，CNN+ViT选题

Tianjin University, Machine Learning and Deep Learning, 2025 spring final assignment.
Repository: [www.github.com/sink-sea/ml-dl-25sp](https://github.com/sink-sea/ml-dl-25sp)

## 实验环境

+ 操作系统：Windows 10
+ Python=3.10.13
+ Pytorch: torch=2.3.0+cu121, torchaudio=2.3.0+cu121, 
torchvision=0.18.0+cu121
+ 硬件环境：Intel Core i5-12500H CPU @ 3.110GHz, 16GB RAM; NVIDIA GeForce RTX 3050 Ti Laptop 4GB GPU
+ 训练设置：Batch-size=128, Epochs=100

## 数据集

+ CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
+ Tiny-ImageNet-200: http://cs231n.stanford.edu/tiny-imagenet-200.zip

## 安装
推荐使用Conda，若没有Conda，可参考[conda安装方式](https://www.anaconda.com/docs/getting-started/miniconda/main#latest-miniconda-installer-links).

We recommend to use Conda, if you don't have Conda installed, you can check out [How to Install Conda](https://www.anaconda.com/docs/getting-started/miniconda/main#latest-miniconda-installer-links).

```bash
# Clone the repo
git clone https://github.com/sink-sea/ml-dl-25sp.git
cd ml-dl-25sp

# Create a virtual env. We use Conda
conda create -n ml-dl python=3.10.8
conda activate ml-dl

# install requirements
pip install -r requirements.txt
```

## 运行方式
+ 直接训练和保存模型
```bash
 python .\main.py --save_model --model_path 'path_to_save' --epochs 100 --plot_accuracy --model 'model_name' --dataset imagenet
```

+ 从本地加载模型
```bash
 python .\main.py --load_model 'path_to_load' --epochs 100  --model 'model_name'
 ```

## 实验结果

| Model | Acc(CIFAR-10) | Acc(ImageNet)
| :---: | :---: | :---:|
| ResNet-18 | 82.16% | 32.30% |
| ResNet(Deep-Fusion) | 81.73% | 33.09% |
| ResNet(Late-Fusion) | 81.54% | 34.04% |
| Pure ViT | 72.63% | 3.02% |
| CoAtNet | 69.88% | 21.25% |
