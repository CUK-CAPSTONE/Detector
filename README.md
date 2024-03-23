## Detector

### Environment Setting
pip install -r requirements.txt

### Build your dataset
Download the appropriate version of ChromeDriver.

Crawler/crawl.py

### Make your checkpoint
FaceDetector/train.py

### Test in Localhost
Flask/app.py

curl -X POST -F "image=@your_image_path" http://localhost:5000/image
