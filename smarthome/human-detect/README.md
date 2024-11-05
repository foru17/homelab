# deepface-fastapi

## Introduction

This is a simple project to detect human faces in an image using deepface and fastapi.

## Requirements

- Python 3.6+
- Debian / MacOS

## Reference

- [deepface](https://github.com/serengil/deepface)

## Install

```bash
pip3 install fastapi
pip3 install "uvicorn[standard]"
pip3 install python-multipart
pip3 install numpy
pip3 install opencv-python-headless
pip3 install retinaface==1.1.1
pip3 install deepface
pip3 install tf-keras
```

## Usage

### Detect and return image file

```bash
curl --location 'http://127.0.0.1:8008/detect_and_return/' \
--form 'file=@"/Users/luolei/Desktop/human-image-example.jpg"'
```

The response will be a image file with faces detected.

### Detect Image and return json

```bash
curl --location 'http://127.0.0.1:8008/detect/' \
--form 'file=@"/Users/luolei/Desktop/human-image-example.jpg"'
```

The response will be a json like below:

```json
{
  "status": "success",
  "faces_detected": 1,
  "faces": [
    {
      "position": {
        "x": 943,
        "y": 1126,
        "w": 131,
        "h": 178
      },
      "confidence": 0.9973875880241394
    }
  ],
  "output_file": "/Users/luolei/Desktop/result_20241105_045005.jpg"
}
```
