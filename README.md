# 数据增广
## 四大数据增广（或称数据增强）的方法分别如下：
1、水平翻转和垂直翻转：将照片水平或垂直翻转，以增加数据的多样性和数量，扩展训练集。目的在于解决平移不变性问题。

2、随机旋转：对照片进行随机旋转，以增加数据的多样性和角度变化，增强模型的鲁棒性。目的在于解决旋转不变性问题。

3、随机裁切：针对原始图像进行不同区域的随机裁剪，以产生不同大小的图像，并将其作为训练样本。目的在于解决尺寸不变性问题。

4、随机色度变换：改变图像的亮度、对比度和饱和度等色调，增加数据的色彩变化，增强模型的鲁棒性，使其在不同光照和颜色条件下具有更好的稳健性。目的在于解决光照复杂性问题。
 
## 该实验选择的图片
![cat](https://user-images.githubusercontent.com/128216499/228540657-44691d0c-e72c-46b8-afb4-fa837c4101e4.jpg)

## 运行train.py代码
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
## 运行traiin.py结果
![image](https://user-images.githubusercontent.com/128216499/228105669-732f916d-8abb-4a1a-be6c-285f1da79ab8.png)

## 运行flip horizontal.py代码（随机水平翻转）
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
import matplotlib.pyplot as pLt
pLt . imshow(augmented_image1 . permute(1, 2, 0))
pLt. show()
```

## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228819448-3e92f5ac-77c2-4c73-93dd-79a41f788881.png)

## 运行rotate.py代码
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
import matplotlib.pyplot as pLt
pLt . imshow(augmented_image1 . permute(1, 2, 0))
pLt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820138-fa30eda4-fe93-4016-b4a4-93aad09e833c.png)

## 运行scale cutting.py代码
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
import matplotlib.pyplot as pLt
pLt . imshow(augmented_image1 . permute(1, 2, 0))
pLt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820614-c238d08d-2120-4b59-939a-4b89f89b2e9e.png)

## 运行data augmentation.py代码
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
import matplotlib.pyplot as pLt
pLt . imshow(augmented_image1 . permute(1, 2, 0))
pLt. show()
```
## 运行结果
![image](https://user-images.githubusercontent.com/128216499/228820969-943d35af-4586-49ef-b1b0-aac6049f3f80.png)

