# Vehicle Detection using YOLOv3 and OpenCV

![Vehicle Detection](https://img.shields.io/badge/Computer%20Vision-Vehicle%20Detection-blue)
![YOLOv3](https://img.shields.io/badge/Model-YOLOv3-brightgreen)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-red)

A real-time vehicle detection system that leverages the YOLOv3 (You Only Look Once) object detection algorithm and OpenCV's Deep Neural Network (DNN) module to identify and track vehicles in images, videos, or live webcam feeds.

## 📋 Features

- Detects vehicles and other objects from the COCO dataset in real-time
- Supports multiple input sources:
  - Image files (`.jpg`, `.png`, etc.)
  - Video files (`.mp4`, `.avi`, etc.)
  - Live webcam feed
- Utilizes pre-trained YOLOv3 model for efficient detection
- Supports both CPU and GPU acceleration for inference
- Draws bounding boxes with class labels and confidence scores
- Displays real-time performance metrics (inference speed, vehicle count)
- Saves processed outputs automatically

## 🛠️ Requirements

- Python 3.x
- OpenCV with DNN support
  - For GPU acceleration: OpenCV built with CUDA support + compatible NVIDIA GPU
- NumPy
- YOLOv3 model files:
  - `yolov3.weights`: Pre-trained weights
  - `yolov3.cfg`: Network configuration
  - `coco.names`: Class names list

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/vehicle-detection-yolo.git
   cd vehicle-detection-yolo
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Download YOLOv3 files:**
   ```bash
   # Download weights file (237MB)
   wget https://pjreddie.com/media/files/yolov3.weights
   
   # Download configuration and class names files
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
   ```

## 🖥️ Usage

### Command syntax:
```bash
python vehicle_detection_yolo.py [--image <path_to_image>] [--video <path_to_video>] [--device <'cpu'/'gpu'>]
```

### Arguments:
- `--image`: (Optional) Path to an input image file
- `--video`: (Optional) Path to an input video file
- `--device`: (Optional) Inference device ('cpu' or 'gpu', defaults to 'cpu')

### Examples:

**Process an image using CPU:**
```bash
python vehicle_detection_yolo.py --image=cars.jpg
```

**Process an image using GPU:**
```bash
python vehicle_detection_yolo.py --image=cars.jpg --device='gpu'
```

**Process a video using CPU:**
```bash
python vehicle_detection_yolo.py --video=traffic.mp4
```

**Process a video using GPU:**
```bash
python vehicle_detection_yolo.py --video=traffic.mp4 --device='gpu'
```

**Use webcam with CPU (default):**
```bash
python vehicle_detection_yolo.py
```

**Use webcam with GPU:**
```bash
python vehicle_detection_yolo.py --device='gpu'
```

## 📊 Output

- Detection results are displayed in a window titled "Deep learning vehicle detection in OpenCV"
- Bounding boxes highlight detected objects with class labels and confidence scores
- Performance metrics shown in the top-left corner:
  - Inference time per frame (ms)
  - Number of vehicles detected
- Processed files are saved as:
  - Images: `<input_name>_yolo_out_py.jpg`
  - Videos: `<input_name>_yolo_out_py.avi`
  - Webcam: `yolo_out_py.avi`

## ⚙️ Configuration

The script uses these default parameters, which can be modified in the source code:

```python
conf_thres = 0.5  # Confidence threshold
iou_thres = 0.4   # Non-Maximum Suppression (NMS) threshold
img_width = 416   # Input width for YOLO network
img_height = 416  # Input height for YOLO network
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv3](https://pjreddie.com/darknet/yolo/) for the object detection algorithm
- [OpenCV](https://opencv.org/) for the computer vision library

 *To access the project's tech report* [click here](https://docs.google.com/document/d/1y6cicVDGyPMFGnKAlqGxWDN4zQk0D9uY9cUfJcdD6Rc/edit?usp=sharing)
 
 ## Authors
- [Mohamed Ahmed](https://github.com/mahmedken)
- [Khaled El Bastawesy](https://github.com/khaledElbastawisy)
