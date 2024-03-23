import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import random
from PIL import Image

import numpy as np
import time
import os
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt



# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './data'

train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)

print('학습 데이터셋 크기:', len(train_datasets))

class_names = train_datasets.classes
print('학습 클래스:', class_names)


def imshow(input, title):
    input = input.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)

    plt.imshow(input)
    plt.title(title)
    plt.show()

iterator = iter(train_dataloader)

inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs)


model = models.resnet34(pretrained=True)
num_features = model.fc.in_features

model.fc = nn.Sequential(     
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_features),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

model = model.to(device)

num_epochs = 30

best_epoch = None
best_loss = 5

''' Train '''
for epoch in tqdm(range(num_epochs)):
    model.train()
    start_time = time.time()
    
    running_loss = 0.
    running_corrects = 0

    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch
        print("best_loss: {:.4f} \t best_epoch: {}".format(best_loss, best_epoch))

os.makedirs('./weight',exist_ok=True)
torch.save(model, './weight/model_best_epoch.pt')

''' Test '''
valid_images = []
valid_dir = data_dir + '/test'

val_folders = glob(valid_dir + '/*')
for val_folder in val_folders:
    image_paths = glob(val_folder + '/*')
    for image_path in image_paths: valid_images.append(image_path)



num = random.randint(0,len(valid_images)-1)
valid_image = valid_images[num]

image = Image.open(valid_image)
image = transforms_test(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    imshow(image.cpu().data[0], title=' 학습 결과 : ' + class_names[preds[0]])

