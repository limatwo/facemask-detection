# Face Mask Detection

A real-time face mask detection system using deep learning that can identify whether a person is wearing a face mask or not through video feed or webcam input.

## 🎯 Overview

This project implements a face mask detection system using MobileNet v2 architecture for efficient and accurate classification. The system can process live video streams and detect faces while classifying whether each person is wearing a mask or not.

## 🔧 Technologies Used

- **Python 3.2**
- **OpenCV** - Computer vision library for image/video processing
- **TensorFlow/Keras** - Deep learning framework
- **MobileNet v2** - Lightweight neural network architecture
- **NumPy** - Numerical computations
- **Jupyter Notebook** - Development environment

## 📁 Project Structure

```
facemask-detection/
├── README.md
├── detect_mask_video.py          # Main video detection script
├── Mobilenet_v2_facemask.ipynb   # Training notebook
├── mask_detector.model_.h5       # Trained mask detection model
├── deploy.prototxt               # Face detection model config
├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detector
└── .ipynb_checkpoints/           # Jupyter checkpoint files
```

## 🚀 Features

- **Real-time Detection**: Process live video feed from webcam
- **High Accuracy**: Uses MobileNet v2 for reliable classification
- **Efficient Processing**: Optimized for real-time performance
- **Face Detection**: Automatically detects faces in the frame
- **Visual Feedback**: Draws bounding boxes with classification results

## 📋 Requirements

```
opencv-python
tensorflow
numpy
imutils
argparse
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/limatwo/facemask-detection.git
cd facemask-detection
```

2. Install required dependencies:
```bash
pip install opencv-python tensorflow numpy imutils argparse
```

## 🎮 Usage

### Video Detection
Run the main detection script:
```bash
python detect_mask_video.py
```

### Jupyter Notebook
Open the training notebook to understand the model development process:
```bash
jupyter notebook Mobilenet_v2_facemask.ipynb
```

## 🧠 Model Architecture

- **Base Model**: MobileNet v2 (pre-trained on ImageNet)
- **Custom Layers**: Added dense layers for binary classification
- **Input Size**: 224x224 pixels
- **Classes**: 2 (With Mask / Without Mask)
- **Face Detection**: SSD MobileNet for face localization

## 📊 Performance

The model achieves high accuracy in detecting face masks with optimized performance for real-time applications. The MobileNet v2 architecture ensures efficient processing while maintaining detection quality.

## 🔄 How It Works

1. **Face Detection**: Uses SSD face detector to locate faces in the frame
2. **Preprocessing**: Extracts and preprocesses face regions
3. **Classification**: Feeds preprocessed faces to the trained model
4. **Visualization**: Draws bounding boxes with confidence scores



## 📄 License

This project is open source and available under the [MIT License](LICENSE).


## 🙏 Acknowledgments

- MobileNet v2 architecture by Google
- OpenCV community for computer vision tools
- TensorFlow team for the deep learning framework

