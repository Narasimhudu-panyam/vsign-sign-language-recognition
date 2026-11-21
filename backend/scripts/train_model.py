import numpy as np
import os
from pathlib import Path
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_DIR / "data" / "landmarks"
MODEL_DIR = BACKEND_DIR / "model"

X = []
y = []

labels = os.listdir(DATA_PATH)
label_map = {label: idx for idx, label in enumerate(labels)}

print("🔄 Loading data...")

for label in labels:
    folder = os.path.join(DATA_PATH, label)

    files = os.listdir(folder)[:200]   # 🔥 limit for speed

    for file in files:
        data = np.load(os.path.join(folder, file))

        if data.shape != (63,):
            continue

        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("✅ Total samples:", len(X))

# SCALE
scaler = StandardScaler()
X = scaler.fit_transform(X)

# SAVE scaler
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, str(MODEL_DIR / "scaler.save"))

# SAVE labels
with open(MODEL_DIR / "labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# MODEL
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🚀 Training...")

model.fit(X_train, y_train, epochs=10, batch_size=16)

model.save(str(MODEL_DIR / "sign_model.h5"))

print("✅ Training completed!")