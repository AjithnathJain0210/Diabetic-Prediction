"""
Module 3A: Fingerprint Image Preprocessing
Purpose: Fixed Skeletonization to resolve 'All Whorl' and high-count errors.
"""
import cv2
import numpy as np
import os
from config import FINGERPRINT_PROCESSED_DIR

def morphological_thinning(image):
    """Guaranteed skeletonization using core OpenCV functions."""
    _, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = img.copy()
    
    while True:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel

def zhang_suen_thinning(binary_image):
    """Zhang-Suen with Morphological fallback."""
    try:
        img = np.where(binary_image > 0, 1, 0).astype(np.uint8)
        thinned = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        result = (thinned * 255).astype(np.uint8)
        if cv2.countNonZero(result) == 0:
            return morphological_thinning(binary_image)
        return result
    except:
        return morphological_thinning(binary_image)

def preprocess_fingerprint(image, patient_id=None, save=False):
    """Refined Pipeline to fix Pattern and Ridge Count errors."""
    
    # Step 0: Enhancement (Tuned CLAHE for your dataset)
    # ClipLimit 3.0 prevents over-sharpening noise
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    # Step 1: Denoising (CRITICAL: Removes salt-and-pepper noise)
    denoised = cv2.medianBlur(enhanced, 5)

    # Step 2: Adaptive Binarization
    # C=7 helps keep ridges thin and separate, preventing false Deltas
    binary = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 7
    )
    
    # Step 3: Morphological Closing (THE WHORL FIX)
    # Fills tiny 'white holes' in the ridges so the skeleton is smooth
    close_kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    # Step 4: Skeletonization
    thinned = zhang_suen_thinning(closed)
    
    # --- AUDIT VIEW ---
    r_view = cv2.resize(image, (250, 350))
    b_view = cv2.resize(closed, (250, 350))
    s_view = cv2.resize(thinned, (250, 350))
    debug_panel = np.hstack((r_view, b_view, s_view))
    
    # Optional: Comment these 3 lines if batch processing thousands of images
    cv2.imshow(f"Debug: {patient_id}", debug_panel)
    print(f"✓ Viewing: {patient_id}. Press any key...")
    cv2.waitKey(1) # Using 1 for faster batch testing, change to 0 for manual check
    # ------------------

    if save and patient_id:
        os.makedirs(FINGERPRINT_PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(FINGERPRINT_PROCESSED_DIR, f"{patient_id}_proc.png")
        cv2.imwrite(output_path, thinned)
    
    return thinned

def preprocess_all_fingerprints(fingerprints_dict, save=True):
    processed = {}
    print("\n" + "="*50)
    print("✓ MODULE 3A: FIXING PATTERN & COUNT LOGIC")
    print("="*50)
    
    for pid, img in fingerprints_dict.items():
        try:
            processed[pid] = preprocess_fingerprint(img, pid, save)
        except Exception as e:
            print(f"✗ Error in {pid}: {str(e)}")
            
    print(f"\n✓ Processed {len(processed)} skeletons successfully.")
    return processed

if __name__ == "__main__":
    from src.acquisition import load_fingerprint_images
    fingerprints = load_fingerprint_images()
    if fingerprints:
        preprocess_all_fingerprints(fingerprints)