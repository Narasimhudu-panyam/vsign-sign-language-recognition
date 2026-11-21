# 🖐 VSign — Sign Language Recognition

Real-time sign language recognition using **Streamlit**, **MediaPipe**, **OpenCV**, and **TensorFlow/Keras**.

## Features
- Webcam gesture classification
- Live WebRTC hand recognition
- Two-user video-call demonstration
- Video landmark extraction and model training

## Project Structure

```text
vsign-sign-language-recognition/
├── frontend/
│   ├── app.py                     # Single-user Streamlit camera app
│   ├── video_app.py               # WebRTC sign recognition
│   └── video_call.py              # Two-user WebRTC demo
├── backend/
│   ├── scripts/
│   │   ├── create_landmark_dataset.py
│   │   └── train_model.py
│   ├── data/
│   │   ├── Video_Dataset/        # Training videos (local, ignored)
│   │   └── landmarks/            # Generated landmark arrays (ignored)
│   ├── model/
│   │   ├── sign_model.h5         # Trained model (ignored)
│   │   ├── labels.pkl            # Label mapping (ignored)
│   │   └── scaler.save           # Scaler (ignored)
│   └── requirements.txt
├── .gitignore
└── README.md
```

**Architecture:** This project uses Streamlit as its UI. The backend folder contains ML training scripts and model data, *not* a separately running FastAPI server. The structure is organizational and preserves the existing technology stack.

## Install

From the repository root, create and activate a Python environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

For macOS/Linux, activate with `source .venv/bin/activate`.

## Prepare training data

Put labeled videos in `backend/data/Video_Dataset/<sign>/`, then from the repository root:

```bash
python backend/scripts/create_landmark_dataset.py
python backend/scripts/train_model.py
```

Training artifacts are written to `backend/model/`. Raw videos, landmarks and trained models are excluded from Git. **Important:** Confirm that landmark preprocessing in extraction and inference matches before evaluating recognition accuracy.

## Run the frontend

After creating or providing `backend/model/sign_model.h5`, `backend/model/labels.pkl`, and `backend/model/scaler.save`:

```bash
streamlit run frontend/app.py
```

Alternate WebRTC modes:

```bash
streamlit run frontend/video_app.py
streamlit run frontend/video_call.py
```

Run the commands from the repository root. Webcam permissions and WebRTC connectivity may be required.

## How recognition works

MediaPipe extracts 21 hand landmarks (63 coordinate values). A saved scaler normalizes features and the Keras model classifies the sign. Confidence thresholds and rolling voting reduce unstable predictions.

## Tech stack
Python, Streamlit, OpenCV, MediaPipe, TensorFlow/Keras, scikit-learn, NumPy, joblib and streamlit-webrtc.

## Note
The source layout and run paths were updated; the application was not executed or trained as part of this repository reorganization.

## Author
Panyam Lakshmi Narasimhudu
