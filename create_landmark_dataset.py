import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

DATA_PATH = Path("data/Video_Dataset")
SAVE_PATH = Path("data/landmarks")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_label_directories(root_path: Path):
    return sorted(path for path in root_path.iterdir() if path.is_dir())


def iter_video_files(label_path: Path):
    return sorted(
        path for path in label_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_PATH.resolve()}")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    print("Starting landmark extraction...")

    total_saved = 0
    total_videos = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        for label_path in iter_label_directories(DATA_PATH):
            save_label_path = SAVE_PATH / label_path.name
            save_label_path.mkdir(parents=True, exist_ok=True)

            print(f"Processing label: {label_path.name}")

            for video_path in iter_video_files(label_path):
                cap = cv2.VideoCapture(str(video_path))

                if not cap.isOpened():
                    print(f"Skipping unreadable video: {video_path}")
                    continue

                total_videos += 1
                frame_index = 0
                saved_for_video = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            coords = []
                            for landmark in hand_landmarks.landmark:
                                coords.extend([landmark.x, landmark.y, landmark.z])

                            output_path = save_label_path / f"{video_path.stem}_{frame_index}.npy"
                            np.save(output_path, np.array(coords, dtype=np.float32))
                            frame_index += 1
                            saved_for_video += 1
                            total_saved += 1

                cap.release()
                print(f"  Saved {saved_for_video} frames from {video_path.name}")

    print(
        f"Landmark dataset created successfully. "
        f"Processed {total_videos} videos and saved {total_saved} landmark files."
    )


if __name__ == "__main__":
    main()