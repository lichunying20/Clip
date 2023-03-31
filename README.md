# 数据增广
## 四大数据增广（或称数据增强）的方法分别如下：
1、水平翻转和垂直翻转：将照片水平或垂直翻转，以增加数据的多样性和数量，扩展训练集。目的在于解决平移不变性问题。

2、随机旋转：对照片进行随机旋转，以增加数据的多样性和角度变化，增强模型的鲁棒性。目的在于解决旋转不变性问题。

3、随机裁切：针对原始图像进行不同区域的随机裁剪，以产生不同大小的图像，并将其作为训练样本。目的在于解决尺寸不变性问题。

4、随机色度变换：改变图像的亮度、对比度和饱和度等色调，增加数据的色彩变化，增强模型的鲁棒性，使其在不同光照和颜色条件下具有更好的稳健性。目的在于解决光照复杂性问题。
 
## 该实验选择的图片
训练所用的照片

![cat](https://user-images.githubusercontent.com/128216499/228540657-44691d0c-e72c-46b8-afb4-fa837c4101e4.jpg)

验证所用的照片

![dog](https://user-images.githubusercontent.com/128216499/228836219-0dd47421-a0ac-42b6-befa-f56f3251ed64.jpg)

## 训练模型（运行train.py）
```python
import argparse
import time
import json
import os

from tqdm import tqdm
from models import *
# from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

    # model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        kwargs = {
            'num_workers': args.num_workers,
            'pin_memory': True,
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        assert net is not None

        self.model = net.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            data, target = data.to(self.device), target.to(self.device)

            # 训练中...
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step
            )

            # 更新进度条
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(),
                    lr
                )
            )
        bar.close()

    def val(self):
        self.model.eval()
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.val_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        validating_loss /= len(self.val_loader)
        print('val >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            validating_loss,
            num_correct,
            len(self.val_loader.dataset),
            100. * num_correct / len(self.val_loader.dataset))
        )

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)

```
## 训练结果
![1E355E02ED7853404B76F69BB4ED3C54](https://user-images.githubusercontent.com/128216499/229102263-f982f5a5-116d-4140-b3a0-ff92efeedcd9.jpg)

## 运行flip horizontal.py（随机水平翻转）
```python
import torch
import torchvision. transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms . Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms . ToTensor(),
])
# 加载图片
image1 = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')
# 对图片进行增广
augmented_image1 = transform(image1)
# 显示增广后的图片
import matplotlib.pyplot as plt
plt . imshow(augmented_image1 . permute(1, 2, 0))
plt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228819448-3e92f5ac-77c2-4c73-93dd-79a41f788881.png)

## 运行rotate.py（随机旋转）
```python
import torch
import torchvision. transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms . Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms . ToTensor(),
])
# 加载图片
image1 = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')

# 对图片进行增广
augmented_image1 = transform(image1)

# 显示增广后的图片
import matplotlib.pyplot as plt
plt . imshow(augmented_image1 . permute(1, 2, 0))
plt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820138-fa30eda4-fe93-4016-b4a4-93aad09e833c.png)

## 运行scale cutting.py（随机缩放裁切，裁切后尺寸256）
```python
import torch
import torchvision. transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms . Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms . ToTensor(),
])
# 加载图片
image1 = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')
# 对图片进行增广
augmented_image1 = transform(image1)
# 显示增广后的图片
import matplotlib.pyplot as plt
plt . imshow(augmented_image1 . permute(1, 2, 0))
plt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820614-c238d08d-2120-4b59-939a-4b89f89b2e9e.png)

## 运行data augmentation.py（三种数据增广相结合）
```python
import torch
import torchvision. transforms as transforms
from PIL import Image

# 定义数据增广函数
transform = transforms . Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms . ToTensor(),
])
# 加载图片
image1 = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')
# 对图片进行增广
augmented_image1 = transform(image1)
# 显示增广后的图片
import matplotlib.pyplot as plt
plt . imshow(augmented_image1 . permute(1, 2, 0))
plt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820969-943d35af-4586-49ef-b1b0-aac6049f3f80.png)

## 运行test.py（训练集数据和验证集数据都进行了数据增广，但两张图片不同时展示）
```python
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据增强变换
transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
])
# 加载训练集图片
train_image = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')

# 对训练集图片进行增广
augmented_train_image = transform(train_image)

# 显示增广后的图片
plt.imshow(augmented_train_image.permute(1, 2, 0))
plt.show()

# 加载验证集图片
val_image = Image.open(r'D:\GitHub_test\classification-basic-sample\data\val\val\dog.jpg')

# 对验证集图片进行增广
augmented_val_image = transform(val_image)

# 显示增广后的图片
plt.imshow(augmented_val_image.permute(1, 2, 0))
plt.show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228834540-55f0a7e3-802e-4994-9af0-2247e254371f.png)
![image](https://user-images.githubusercontent.com/128216499/228834751-242cd548-5bab-442b-a473-081385061291.png)

## 运行train_val.py（训练集数据和验证集数据都进行了数据增广，但两张图片可以同时展示）
```python
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据增强变换
transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
])

# 加载训练集图片和验证集图片
train_image = Image.open(r'D:\GitHub_test\classification-basic-sample\data\train\train\cat.jpg')
val_image = Image.open(r'D:\GitHub_test\classification-basic-sample\data\val\val\dog.jpg')

# 对训练集图片和验证集图片进行增广
augmented_train_image = transform(train_image)
augmented_val_image = transform(val_image)

# 显示增广后的图片
fig, axs = plt.subplots(1, 2)
axs[0].imshow(augmented_train_image.permute(1, 2, 0))
axs[1].imshow(augmented_val_image.permute(1, 2, 0))
plt.show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228833428-7e9ef645-87ce-43a6-b869-2c3bf5851d63.png)

## 对数据集进行数据增广（运行data set.py）
```python
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

data_dir = r'D:\GitHub_test\classification-basic-sample\cats_and_dogs_dataset\train'

transform = transforms.Compose([
    transforms.RandomResizedCrop(100, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
])

data = dset.ImageFolder(root=data_dir, transform=transform)

fig = plt.figure(figsize=(10, 10))

for i, (img, label) in enumerate(data):
    print("Image {}: Label {}".format(i, label))

    # Convert tensor to numpy array
    img = img.numpy().transpose((1, 2, 0))

    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()

    if i == 40:
        break

plt.show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/229121694-b3262fba-e9f8-4cc8-ba25-538aaa4869e9.png)

![image](https://user-images.githubusercontent.com/128216499/229126962-a869ade9-23ae-46fc-8974-fd9b76a60850.png)

## 个人总结
1、进行模型训练时（运行train.py），可能会遇到这个错误，需要增加训练集和验证集中图片的数量。
```python
ZeroDivisionError: float division by zero
```
2、flip horizontal.py、rotate.py和scale cutting.py分别对图像进行随机水平翻转、随机旋转和随机缩放裁切，而data augmentation.py三种数据增广方式结合在一起，最后的图像也包含了这三种形式。

3、test.py和train_val.py都对训练图片和验证图片进行了数据增广，不同的是test.py得到的图片为两张且不能同时出现，而train_val.py得到的图片为一张且包含了训练图片和验证图片。

4、关于代码的更改：

（1）、随机水平翻转：RandomHorizontalFlip()会将图片随机水平翻转,而p=0.5指的是概率为50%。
```python
    transforms.RandomHorizontalFlip(p=0.5),
```
（2）、随机旋转：RandomRotation()会对图片进行随机旋转，而degree(-10，10)表示旋转角度的范围为-10度到10度之间，角度范围可以根据自己的需要进行更改。
```python
    transforms.RandomRotation(degrees=(-10, 10)),
```
（3）、随机缩放裁切：RandomResizedCrop()会将图片随机剪裁为256x256大小的图像，然后缩放到一个比例模型的范围内。通过scale参数设置变换的尺度范围，通过ratio参数控制宽高比的范围，这里设置为（1.0,1.0）即不改变宽高比。
```python
transforms.RandomResizedCrop(256, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
```

5、在进行数据集的数据增广时，图片都是一张一张出现的，删除一张，下一张才会出现。运行结果中的Label表示每个图片的标签，每个文件夹下都有20张图片，所以标签为0和1各有20张（'cats'文件夹下的所有图像的标签都为0，'dogs'文件夹下的图像标签都为1）。
