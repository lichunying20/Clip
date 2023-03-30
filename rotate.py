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