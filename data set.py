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