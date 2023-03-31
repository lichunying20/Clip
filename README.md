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

## 个人总结
1、运行train.py时，始终运行不出来结果，看不到最后图片，只好放弃，换个代码使用。

2、flip horizontal.py、rotate.py和scale cutting.py分别对图像进行随机水平翻转、随机旋转和随机缩放裁切，而data augmentation.py三种数据增广方式结合在一起，最后的图像也包含了这三种形式。

3、test.py和train_val.py都对训练图片和验证图片进行了数据增广，不同的是test.py得到的图片为两张且不能同时出现，而train_val.py得到的图片为一张且包含了训练图片和验证图片。

4、关于代码的更改：

（1）、随机水平翻转，RandomHorizontalFlip()会将图片随机水平翻转,而p=0.5指的是概率为50%。

（2）、随机旋转，RandomRotation()会对图片进行随机旋转，而degree(-10，10)表示旋转角度的范围为-10度到10度之间，角度范围可以根据自己的需要进行更改。
   
（3）、随机缩放裁切，RandomResizedCrop()会将图片随机剪裁为256x256大小的图像，然后缩放到一个比例模型的范围内。通过scale参数设置变换的尺度范围，通过ratio参数控制宽高比的范围，这里设置为（1.0,1.0）即不改变宽高比。
   
