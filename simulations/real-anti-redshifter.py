import cv2
import numpy as np

img = cv2.imread("galaxy-image.png").astype(np.float32)
brightnessLevel = int(input("How much do you want to increase the brightness for the thresholding? (Enter a number, e.g., 10 for 10x brightness) Ideally experiment with all values ——— the brighter the more visible dim objects are ——— but sometimes high brightness can glitch out in the thresholding step. If you see glitchy black spots, try reducing the brightness: "))

b, g, r = cv2.split(img)
bg_b = cv2.GaussianBlur(b, (0, 0), 80)
bg_g = cv2.GaussianBlur(g, (0, 0), 80)
bg_r = cv2.GaussianBlur(r, (0, 0), 80)

b2 = b - bg_b
g2 = g - bg_g
r2 = r - bg_r

def norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

b2 = norm(b2).astype(np.uint8)
g2 = norm(g2).astype(np.uint8)
r2 = norm(r2).astype(np.uint8)


out = cv2.merge([b2, g2, r2])
hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

s = cv2.multiply(s, 1.4)  
v = cv2.multiply(v, 2)  
enhanced_hsv = cv2.merge([h, s, v])
final_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
b,g,r = cv2.split(final_bgr)
r = cv2.multiply(r, 0.8)
final_bgr = cv2.merge([b, g, r])
usedimg = final_bgr.copy()
TenxBrightness =cv2.cvtColor(usedimg, cv2.COLOR_BGR2HSV)
h , s ,v = cv2.split(TenxBrightness)
v = cv2.multiply(v, brightnessLevel)
y = cv2.merge([h, s, v])
imageUsedForThresholding = cv2.cvtColor(y, cv2.COLOR_HSV2BGR)

gray = cv2.cvtColor(imageUsedForThresholding, cv2.COLOR_BGR2GRAY)


ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


thresholded_img = cv2.bitwise_and(imageUsedForThresholding, imageUsedForThresholding, mask=mask)


cv2.imshow("Original Processed", final_bgr)
cv2.imshow("Threshold Mask", mask)
cv2.imshow("Isolated Objects Better quality", thresholded_img)
cv2.imshow("User Chosen Brightness", imageUsedForThresholding)

cv2.waitKey(0)
cv2.destroyAllWindows()