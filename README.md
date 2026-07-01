# 🖐 VSign - Sign Language Recognition

A real-time sign language recognition system using MediaPipe hand detection and a deep neural network model built with TensorFlow/Keras.

## 📋 Overview

VSign is a comprehensive project that includes:
- **Video dataset processing** with landmark extraction
- **Neural network training** for sign language classification
- **Real-time recognition** with Streamlit web interface
- **Multi-user video streaming** capabilities

## 🎯 Features

- 🎥 Real-time hand detection and tracking using MediaPipe
- 🧠 Deep learning model for sign language classification
- 📊 Confidence-based prediction with voting mechanism
- ⚡ Optimized performance with frame skipping
- 🌐 Web interface built with Streamlit
- 👥 Multi-user video call support
- 📝 Normalized landmark extraction (relative to hand base)

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit app for single user
├── video_app.py                # Live video stream app
├── video_call.py               # Multi-user video call app
├── create_landmark_dataset.py  # Extract landmarks from video dataset
├── train_model.py              # Train the sign language model
├── README.md                   # This file
└── model/
    ├── sign_model.h5           # Trained TensorFlow model
    ├── labels.pkl              # Label mapping
    └── scaler.save             # StandardScaler for normalization
```

## 🔧 Requirements

```bash
pip install streamlit
pip install opencv-python
pip install tensorflow
pip install mediapipe
pip install numpy
pip install scikit-learn
pip install joblib
pip install streamlit-webrtc
```

## 📊 Data Pipeline

### 1. Dataset Structure
```
data/
├── Video_Dataset/
│   ├── sign_A/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   └── sign_B/
│       └── video3.mp4
└── landmarks/  (Generated)
    ├── sign_A/
    │   ├── video1_0.npy
    │   └── video1_1.npy
    └── sign_B/
        └── video3_0.npy
```

### 2. Extract Landmarks
```bash
python create_landmark_dataset.py
```
This script:
- Processes all videos in `data/Video_Dataset/`
- Extracts hand landmarks using MediaPipe
- Saves normalized coordinates as `.npy` files in `data/landmarks/`

### 3. Train Model
```bash
python train_model.py
```
This script:
- Loads landmark data from `data/landmarks/`
- Normalizes features using StandardScaler
- Trains a 3-layer Dense neural network
- Saves model, scaler, and label mappings

## 🚀 Running the Application

### Single User Mode
```bash
streamlit run app.py
```
- Click "▶ Start Camera" to begin
- Make sign language gestures
- Recognized signs appear at the bottom

### Live Video Stream
```bash
streamlit run video_app.py
```
- WebRTC-based real-time streaming
- Displays recognized signs on video

### Multi-User Video Call
```bash
streamlit run video_call.py
```
- Two-user simultaneous recognition
- Side-by-side video streams
- Real-time sign detection for both users

## 🤖 Model Architecture

```python
Input: 63 features (21 landmarks × 3 coordinates)
  ↓
Dense(128, relu) - 128 neurons
  ↓
Dense(64, relu) - 64 neurons
  ↓
Dense(num_classes, softmax) - Classification layer
```

### Model Specifications
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 16
- **Input Features**: 63 (normalized hand landmarks)

## 🎯 Recognition Pipeline

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time
2. **Normalization**: Coordinates are normalized relative to the hand's base point
3. **Scaling**: Features are standardized using the fitted scaler
4. **Prediction**: Neural network predicts the sign class
5. **Voting**: Last 5 predictions are collected (confidence > 0.8)
6. **Final Decision**: Mode of 5 predictions determines final sign
7. **Sentence Building**: Signs added with 0.8s time gap to prevent duplicates

## 📈 Performance Optimizations

- **Confidence Threshold**: 0.8 for prediction acceptance
- **Frame Skipping**: Process every 3rd frame in video mode
- **Voting Mechanism**: Requires 3+ votes from 5 predictions
- **Deque Buffer**: Fixed-size queue (maxlen=5) for efficient voting
- **Time-based Filtering**: 0.8s minimum gap between recognized signs

## 🔍 Troubleshooting

### Model not found
- Ensure you've run `train_model.py` first
- Check that `model/` directory contains all required files

### Camera not working
- Verify webcam permissions
- Check that cv2.VideoCapture(0) works on your system

### Low recognition accuracy
- Ensure good lighting conditions
- Keep hand fully visible in frame
- Make clear, distinct hand gestures

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Panyam Lakshmi Narasimhudu**

## 🤝 Contributing

Contributions are welcome! Feel free to fork and submit pull requests.
