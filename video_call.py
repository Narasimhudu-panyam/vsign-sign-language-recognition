import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import joblib
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model("model/sign_model.h5")

with open("model/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

labels = {v: k for k, v in label_map.items()}
scaler = joblib.load("model/scaler.save")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.pred_queue = deque(maxlen=5)
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # 🔥 SKIP FRAMES (VERY IMPORTANT FOR SPEED)
        if self.frame_count % 3 != 0:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                coords = []
                base = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    coords.extend([
                        lm.x - base.x,
                        lm.y - base.y,
                        lm.z - base.z
                    ])

                coords = np.array(coords).reshape(1, -1)
                coords = scaler.transform(coords)

                pred = model.predict(coords, verbose=0)
                prob = np.max(pred)
                class_id = np.argmax(pred)

                if prob > 0.75:
                    self.pred_queue.append(class_id)

                if len(self.pred_queue) == 5:
                    final_class = max(set(self.pred_queue), key=self.pred_queue.count)

                    if self.pred_queue.count(final_class) >= 3:
                        word = labels[final_class]

                        cv2.putText(
                            img,
                            word,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

        return img


st.title("🎥 2-User Sign Language Video Demo")

col1, col2 = st.columns(2)

with col1:
    st.subheader("User A")
    webrtc_streamer(key="user1", video_processor_factory=SignProcessor)

with col2:
    st.subheader("User B")
    webrtc_streamer(key="user2", video_processor_factory=SignProcessor)