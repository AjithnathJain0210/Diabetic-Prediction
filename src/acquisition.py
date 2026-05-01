"""
Module 1: Data Acquisition
Purpose: Load clinical data and fingerprint images
"""
import pandas as pd
import cv2
import os
import glob
from config import CLINICAL_DATA_PATH, FINGERPRINT_RAW_DIR


def load_clinical_data(filepath=CLINICAL_DATA_PATH):
    """
    Load clinical data from Excel file
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        DataFrame with clinical data
    """
    try:
        df = pd.read_excel(filepath)
        print(f"✓ Loaded {len(df)} clinical records")
        print(f"✓ Features: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: Clinical data file not found at {filepath}")
        return None
    except Exception as e:
        print(f"✗ Error loading clinical data: {str(e)}")
        return None


def load_fingerprint_images(image_dir=FINGERPRINT_RAW_DIR):
    """
    Load all fingerprint images from directory
    
    Args:
        image_dir: Directory containing fingerprint images
        
    Returns:
        Dictionary mapping patient_id to image array
    """
    images = {}
    supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(image_dir, fmt)))
    
    if not image_files:
        print(f"✗ No fingerprint images found in {image_dir}")
        return images
    
    for img_path in image_files:
        # Extract patient_id from filename (e.g., patient_001.png -> 001)
        filename = os.path.basename(img_path)
        patient_id = os.path.splitext(filename)[0]
        
        # Handle processed filenames (e.g., "101_proc" -> "101")
        if patient_id.endswith('_proc'):
            patient_id = patient_id[:-5]
        
        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            images[patient_id] = img
        else:
            print(f"✗ Failed to load image: {filename}")
    
    print(f"✓ Loaded {len(images)} fingerprint images")
    return images


def load_single_fingerprint(filepath):
    """
    Load a single fingerprint image for prediction
    
    Args:
        filepath: Path to fingerprint image
        
    Returns:
        Grayscale image array
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from {filepath}")
    return img


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Acquisition Module...")
    print("-" * 50)
    
    clinical_df = load_clinical_data()
    if clinical_df is not None:
        print(f"\nClinical Data Shape: {clinical_df.shape}")
        print(f"First 5 rows:\n{clinical_df.head()}")
    
    fingerprints = load_fingerprint_images()
    if fingerprints:
        print(f"\nLoaded fingerprints for: {list(fingerprints.keys())[:5]}...")
