import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import joblib
import time
from collections import deque

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model("model/sign_model.h5")

with open("model/labels.pkl", "rb") as f:
    label_map = pickle.load(f)

labels = {v: k for k, v in label_map.items()}
scaler = joblib.load("model/scaler.save")

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- UI ---------------- #
st.title("🖐 VSign - Sign Language Recognition")

run = st.button("▶ Start Camera")
stop = st.button("⛔ Stop")

frame_placeholder = st.empty()
text_placeholder = st.empty()

# ---------------- MAIN ---------------- #
if run:
    cap = cv2.VideoCapture(0)

    sentence = []
    last_word = ""
    last_time = time.time()

    pred_queue = deque(maxlen=5)   # 🔥 Faster response

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        word = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
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

                # 🔥 BALANCED CONFIDENCE
                if prob > 0.8:
                    pred_queue.append(class_id)

                # 🔥 FAST + STABLE VOTING
                if len(pred_queue) == 5:
                    final_class = max(set(pred_queue), key=pred_queue.count)

                    if pred_queue.count(final_class) >= 3:
                        word = labels[final_class]

        else:
            pred_queue.clear()

        # 🔥 FASTER SENTENCE UPDATE
        if word != "" and word != last_word and (time.time() - last_time) > 0.8:
            sentence.append(word)
            last_word = word
            last_time = time.time()

        # SHOW FRAME
        frame_placeholder.image(frame, channels="BGR")

        # SUBTITLE UI
        text_placeholder.markdown(
            f"""
            <div style="
                position:fixed;
                bottom:20px;
                left:50%;
                transform:translateX(-50%);
                background:black;
                color:white;
                padding:10px 20px;
                border-radius:10px;
                font-size:24px;
                font-weight:bold;
            ">
            {' '.join(sentence[-5:])}
            </div>
            """,
            unsafe_allow_html=True
        )

        if stop:
            break

    cap.release()