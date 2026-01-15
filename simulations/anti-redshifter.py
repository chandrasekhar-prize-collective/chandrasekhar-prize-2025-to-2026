# HAHA THATS WHAT I CALL GLASS PANELS
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def full_galaxy_correction(img_path, out_path):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)

 
    flat = arr.reshape(-1, 3)
    low_vals = np.percentile(flat, 2, axis=0)
    offset = low_vals.mean()
    arr1 = arr - offset
    arr1 = np.maximum(arr1, 0)


    arr1 = arr1 / np.max(arr1) * 255.0

 
    lab = cv2.cvtColor(arr1.astype(np.uint8), cv2.COLOR_RGB2LAB)
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
    Image.fromarray(out).save(out_path)

    return img, out

input_path = "galaxy_image.png"
output_path = "galaxy_corrected.png"

orig, corrected = full_galaxy_correction(input_path, output_path)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(orig)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Processed")
plt.imshow(corrected)
plt.axis("off")

plt.tight_layout()
plt.show()