import kagglehub
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import sys

def setup_directories():
    """Ensure all required directories exist"""
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed", 
        base_dir / "models" / "trained_models",
        base_dir / "models" / "optimized_models",
        base_dir / "models" / "training_history",
        base_dir / "logs"
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory ready: {dir_path}")
    
    return base_dir

def download_and_preprocess_fer2013():
    """Download and preprocess FER2013 dataset for emotion recognition"""
    print("🚀 Starting FER2013 data preprocessing...")
    
    # Setup directories
    base_dir = setup_directories()
    raw_data_dir = base_dir / "data" / "raw"
    
    try:
        # Download dataset
        print("📥 Downloading FER2013 dataset from Kaggle...")
        fer_path = kagglehub.dataset_download("deadskull7/fer2013")
        print(f"✅ Dataset downloaded to: {fer_path}")

        # Find CSV file
        fer_csv = Path(fer_path) / "fer2013.csv"
        if not fer_csv.exists():
            # Try alternative path
            fer_csv = Path(fer_path) / "archive" / "fer2013.csv"
        
        print(f"📄 Loading CSV from: {fer_csv}")
        fer_df = pd.read_csv(fer_csv)
        print(f"📊 Total samples: {len(fer_df):,}")

        # Preprocess in batches to avoid memory issues
        batch_size = 10000
        X_batches = []
        y_batches = []
        
        print("🔄 Preprocessing images...")
        for start_idx in range(0, len(fer_df), batch_size):
            end_idx = min(start_idx + batch_size, len(fer_df))
            batch_df = fer_df.iloc[start_idx:end_idx]
            
            X_batch = []
            y_batch = []
            
            for _, row in batch_df.iterrows():
                # Convert pixel string to 48x48 array
                pixels = np.array(row['pixels'].split(), dtype='float32').reshape(48, 48)
                # Convert to 3 channels
                pixels = np.stack([pixels] * 3, axis=-1)
                # Normalize
                pixels /= 255.0
                X_batch.append(pixels)
                y_batch.append(row['emotion'])
            
            X_batches.append(np.array(X_batch))
            y_batches.extend(y_batch)
            
            print(f"   Processed {end_idx}/{len(fer_df)} images...")
        
        # Combine batches
        X = np.concatenate(X_batches, axis=0)
        y = to_categorical(y_batches, num_classes=7)
        
        print(f"✅ Preprocessing complete!")
        print(f"   Images shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")

        # Save files
        x_path = raw_data_dir / "fer_X.npy"
        y_path = raw_data_dir / "fer_y.npy"
        
        print(f"💾 Saving files...")
        np.save(str(x_path), X)
        np.save(str(y_path), y)
        
        # Verify
        if x_path.exists() and y_path.exists():
            file_size_x = x_path.stat().st_size / (1024 * 1024)  
            file_size_y = y_path.stat().st_size / (1024 * 1024)  
            print(f"✅ Files saved successfully!")
            print(f"   fer_X.npy: {file_size_x:.1f} MB")
            print(f"   fer_y.npy: {file_size_y:.1f} MB")
        else:
            print("❌ Error: Files were not created")
            
        return X, y
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Check internet connection")
        print("   - Ensure kagglehub is installed: pip install kagglehub")
        print("   - Try running as administrator")
        raise

if __name__ == "__main__":
    download_and_preprocess_fer2013()