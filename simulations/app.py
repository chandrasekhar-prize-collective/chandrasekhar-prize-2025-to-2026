import wlrgb
import math
import cv2 as cv
import numpy as np

img = cv.imread('singelpix.png')

distance_galaxy = float(input("Distance to galaxy in m(use scientific e notation):"))
observed_wavelength_galaxy = float(input("Observed wavelength in nm:"))

velocity_galaxy = distance_galaxy * 2.23e-18
redshift_galaxy = (1 + (velocity_galaxy / 3e8)) * (1 / (math.sqrt(1 - (velocity_galaxy ** 2) / ((3e8) ** 2)))) - 1
emitted_wavelength_galaxy = (observed_wavelength_galaxy) / (redshift_galaxy + 1)
rgb_val_galaxy = wlrgb.wavelength_to_rgb(emitted_wavelength_galaxy)

print(f"The RGB value of the pixel in question should be {rgb_val_galaxy}")

b, g, r = cv.split(img)

r = r + rgb_val_galaxy[0]
g = g + rgb_val_galaxy[1]
b = b + rgb_val_galaxy[2]
cv.merge((b,g,r), img)
cv.imshow('Corrected Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
