import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from stress_model import StressDetectionModel, make_representative_gen_from_numpy  # import your class

# ==========================
# Paths
# ==========================
data_dir = Path("data/fer2013_preprocessed")
X_path = data_dir / "fer_X.npy"
y_path = data_dir / "fer_y.npy"  # mapped to stress/fatigue/anomaly or multi-class

out_dir = Path("models/trained_models")
out_dir.mkdir(parents=True, exist_ok=True)

# ==========================
# Load FER2013 data
# ==========================
X = np.load(X_path)  # shape (N, H, W, 1) or (N, H, W, 3)
y_raw = np.load(y_path)  # shape (N, 7) if one-hot 7 emotions

# Example: Map 7 FER classes → binary stress/fatigue/anomaly (simple heuristic)
# 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
y_stress = ((y_raw[:,0] + y_raw[:,2] + y_raw[:,4]) > 0).astype(np.float32).reshape(-1,1)  # Angry, Fear, Sad → Stress
y_fatigue = ((y_raw[:,4] + y_raw[:,6]) > 0).astype(np.float32).reshape(-1,1)           # Sad, Neutral → Fatigue
y_anomaly = ((y_raw[:,1] + y_raw[:,5]) > 0).astype(np.float32).reshape(-1,1)           # Disgust, Surprise → Anomaly

y_labels = {
    'stress': y_stress,
    'fatigue': y_fatigue,
    'anomaly': y_anomaly
}

# ==========================
# Train / Validation split
# ==========================
split = int(0.8 * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train = {k:v[:split] for k,v in y_labels.items()}
y_val = {k:v[split:] for k,v in y_labels.items()}

# ==========================
# Initialize model
# ==========================
img_size = X.shape[1]  # assumes square images
sd = StressDetectionModel((img_size, img_size, X.shape[-1]))
sd.build_temporal_attention_model()
sd.compile_model(lr=1e-3)

# Optional: QAT
# sd.quantize_model()

# ==========================
# Callbacks
# ==========================
ckpt = out_dir / 'stress_best.h5'
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor='val_loss', save_best_only=True)
]

# ==========================
# Training
# ==========================
history = sd.train(
    X_train, y_train,
    X_val, y_val,
    epochs=20,
    batch_size=32,
    callbacks_list=callbacks_list
)

# ==========================
# Save Keras model
# ==========================
keras_path = out_dir / 'stress_detector_final.h5'
sd.model.save(str(keras_path))
print(f"✅ Saved Keras model to: {keras_path}")

# ==========================
# Export TFLite
# ==========================
rep_gen = make_representative_gen_from_numpy(X_train, num_steps=200)
tflite_path = out_dir / 'stress_detector_quant.tflite'
sd.save_as_tflite(str(tflite_path), representative_gen=rep_gen, full_integer=True)
