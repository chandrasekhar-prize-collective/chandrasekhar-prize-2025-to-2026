# HAHA THATS WHAT I CALL GLASS PANELS
import numpy as np
import cv2
import os
import sys

def full_galaxy_correction(img_path, out_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Could not open image. Check the path!")
    
# OpenCV loads in BGR, so we use cv2.cvtColor to get RGB
    arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

 
    flat = arr.reshape(-1, 3)
    low_vals = np.percentile(flat, 2, axis=0)
    offset = low_vals.mean()
    arr1 = arr - offset
    arr1 = np.maximum(arr1, 0)


    arr1 = arr1 / np.max(arr1) * 255.0

 
    img_uint8 = np.clip(arr1, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    
    L, A, B = lab[:,:,0].astype(np.float32), lab[:,:,1].astype(np.float32), lab[:,:,2].astype(np.float32)

    grad_L = np.sqrt(
        cv2.Sobel(L, cv2.CV_32F, 1, 0, 3)**2 +
        cv2.Sobel(L, cv2.CV_32F, 0, 1, 3)**2
    )
    grad_C = np.sqrt(
        cv2.Laplacian(A, cv2.CV_32F)**2 +
        cv2.Laplacian(B, cv2.CV_32F)**2
    )

    mask_artifact = (grad_L > 18) & (grad_C > 14)
    A_blur = cv2.GaussianBlur(A, (5,5), 0)
    B_blur = cv2.GaussianBlur(B, (5,5), 0)
    A[mask_artifact] = A_blur[mask_artifact]
    B[mask_artifact] = B_blur[mask_artifact]

    lab = np.stack([L, A, B], axis=2)
    arr2 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

   
    lum = 0.2126*arr2[:,:,0] + 0.7152*arr2[:,:,1] + 0.0722*arr2[:,:,2]
    bg_floor = np.percentile(lum, 3)
    core_cut = np.percentile(lum, 97)

    disk_mask = (lum > bg_floor + 6) & (lum < core_cut)

   
    warm = np.array([225, 205, 165], dtype=np.float32)
    arr2[disk_mask] *= 1.25
    arr2[disk_mask] = 0.7 * arr2[disk_mask] + 0.3 * warm

    out = np.clip(arr2, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)

    return img_bgr, out


if __name__ == "__main__":
    
    print("Anti-Redshifter Tool")
    print("Paste the path to your image and press Enter.")
    sys.stdout.flush() 
    
    input_path = input("Path: ").strip().replace("'", "").replace('"', '')

    if not os.path.exists(input_path):
        print(f"Error: Could not find file at {input_path}")
        input("Press Enter to close...")
        sys.exit()

    
    file_name, file_ext = os.path.splitext(input_path)
    output_path = file_name + "-antiredshifted.png"

    print("Processing... please wait...")
    
    try:
        orig, corrected = full_galaxy_correction(input_path, output_path)
        print(f"SUCCESS! Saved to: {output_path}")
    except Exception as e:
        print(f"FAILED with error: {e}")

    input("\nProcessing complete. Press Enter to exit.")