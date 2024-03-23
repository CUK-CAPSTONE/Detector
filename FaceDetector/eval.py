import torch
import torchvision
from torchvision import models, datasets, transforms
import os
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("./weight/model_best_epoch.pt", map_location=torch.device('cpu'))

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
train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)
class_names = train_datasets.classes

image = "./static/2.jpg"
image = Image.open(image)
plt.imshow(image)

image = transforms_test(image).unsqueeze(0).to(device)

predictions = []
with torch.no_grad():
    model.eval()
    outputs = model(image)
    outputs = torch.exp(outputs)
    topk, topclass = outputs.topk(3, dim=1)

    for i in range(3):
        predictions.append([class_names[topclass.cpu().numpy()[0][i]],
                            topk.cpu().numpy()[0][i]])

    print(predictions)


