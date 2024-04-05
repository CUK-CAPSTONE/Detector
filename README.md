## Detector

### Environment Setting 
1. python -m venv detect
2. source detect/Scripts/activate
3. pip install -r requirements.txt

### Build your dataset
Download the appropriate version of ChromeDriver.

Crawler/crawl.py

### Make your checkpoint
FaceDetector/train.py

### Test in Localhost
1. Flask/app.py
2. curl -X POST -H "Content-Type: application/json" -d "{\"image_url\":\"https://example.com/path/to/your/image.jpg\"}" http://localhost:5000/image
