import kagglehub
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Step 1: Download FER2013 dataset
# -------------------------------
fer_path = kagglehub.dataset_download("deadskull7/fer2013")
print("FER2013 dataset path:", fer_path)

# CSV file path
fer_csv = os.path.join(fer_path, "fer2013.csv")

# -------------------------------
# Step 2: Load CSV
# -------------------------------
fer_df = pd.read_csv(fer_csv)
print("Total samples:", len(fer_df))

# -------------------------------
# Step 3: Preprocess images and labels
# -------------------------------
X, y = [], []

for idx, row in fer_df.iterrows():
    # Convert pixel string to 48x48 array
    pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
    # Convert grayscale to 3 channels for CNN compatibility
    pixels = np.stack([pixels]*3, axis=-1)
    # Normalize to [0,1]
    pixels /= 255.0
    X.append(pixels)
    y.append(row['emotion'])

X = np.array(X)
y = to_categorical(y, num_classes=7)  # 7 emotions

print("Processed images shape:", X.shape)
print("Processed labels shape:", y.shape)

# -------------------------------
# Step 4: Save preprocessed arrays
# -------------------------------
np.save('fer_X.npy', X)
np.save('fer_y.npy', y)

print("FER2013 preprocessed and saved as 'fer_X.npy' and 'fer_y.npy'")
