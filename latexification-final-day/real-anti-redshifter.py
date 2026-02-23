import cv2
import numpy as np


img = cv2.imread("galaxy_image.png").astype(np.float32)


b, g, r = cv2.split(img)


bg_b = cv2.GaussianBlur(b, (0, 0), 80)
bg_g = cv2.GaussianBlur(g, (0, 0), 80)
bg_r = cv2.GaussianBlur(r, (0, 0), 80)


b2 = b - bg_b
g2 = g - bg_g
r2 = r - bg_r

def norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)

b2 = norm(b2)
g2 = norm(g2)
r2 = norm(r2)


out = cv2.merge([b2, g2, r2]).astype(np.uint8)

img = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img)

s = cv2.multiply(s, 1.5)  
v = cv2.multiply(v, 1.5)  
img = cv2.merge([h, s, v])
out = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

cv2.imshow("new img", out)
cv2.waitKey(0)
cv2.destroyAllWindows()