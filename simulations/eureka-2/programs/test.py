import os
import sys

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' does not exist.")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV failed to load '{path}'.")

    return img.astype(np.float32)


inputImg = input(
    "Enter the name of the image you want to process (e.g., img.png): "
).strip()

if not inputImg:
    print("You did not enter an image name.")
    sys.exit(1)

image_path = f"Images/{inputImg}"

try:
    img = load_image(image_path)
except Exception as e:
    print(f"Error loading '{image_path}': {e}")
    print("Attempting fallback image 'img.png'...")

    try:
        img = load_image("Images/img.png")
    except Exception as e2:
        print(f"Fallback image failed: {e2}")
        sys.exit(1)


try:
    brightnessLevel = int(input("Brightness multiplier (e.g., 10): "))
    if brightnessLevel <= 0:
        raise ValueError("Brightness must be positive.")
except ValueError as e:
    print(f"Invalid brightness input: {e}")
    sys.exit(1)

print("-" * 120)


b, g, r = cv2.split(img)

bg_b = cv2.GaussianBlur(b, (0, 0), 80)
bg_g = cv2.GaussianBlur(g, (0, 0), 80)
bg_r = cv2.GaussianBlur(r, (0, 0), 80)

b2 = b - bg_b
g2 = g - bg_g
r2 = r - bg_r


def norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)  # type:ignore


b2 = norm(b2).astype(np.uint8)
g2 = norm(g2).astype(np.uint8)
r2 = norm(r2).astype(np.uint8)

out = cv2.merge([b2, g2, r2])

hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

s = cv2.multiply(s, np.array([1.4]))
v = cv2.multiply(v, np.array([2]))

enhanced_hsv = cv2.merge([h, s, v])
final_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

b, g, r = cv2.split(final_bgr)
r = cv2.multiply(r, np.array([0.8]))
final_bgr = cv2.merge([b, g, r])

usedimg = final_bgr.copy()

TenxBrightness = cv2.cvtColor(usedimg, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(TenxBrightness)

v = cv2.multiply(v, np.array([brightnessLevel]))

y = cv2.merge([h, s, v])
imageUsedForThresholding = cv2.cvtColor(y, cv2.COLOR_HSV2BGR)

gray = cv2.cvtColor(imageUsedForThresholding, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

thresholded_img = cv2.bitwise_and(
    imageUsedForThresholding, imageUsedForThresholding, mask=mask
)


cv2.imshow("Original Processed", final_bgr)
cv2.imshow("Threshold Mask", mask)
cv2.imshow("Isolated Objects Better quality", thresholded_img)
cv2.imshow("User Chosen Brightness", imageUsedForThresholding)

cv2.waitKey(0)
cv2.destroyAllWindows()
