from PIL import Image
import torch

from torchvision import datasets, models, transforms
import os
import io
import datetime

from flask import Flask, jsonify, request
from flask import jsonify

app = Flask(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '../FaceDetector/data'
train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)
class_names = train_datasets.classes

model = torch.load("./weight/model_best_epoch.pt", map_location=torch.device('cpu'))

def get_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transforms_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        outputs = torch.exp(outputs)
        topk, topclass = outputs.topk(3, dim=1)

        predictions = {}

        for i in range(3):
            predictions[class_names[topclass.cpu().numpy()[0][i]]] = float(topk.cpu().numpy()[0][i])

    return predictions

@app.route('/image', methods=['POST'])
def upload_image_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['image']

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    jpeg_image_bytes = buffer.getvalue()

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"./history/{timestamp}.png"

    result = get_prediction(image_bytes=jpeg_image_bytes)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port = 5000)