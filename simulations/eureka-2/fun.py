import cv2 as cv
import numpy as np

video = cv.VideoCapture(0)
ret, frame = video.read()
while True:
    ret, frame = video.read()
    if not ret:
        break
    y = frame.copy()
    y = cv.cvtColor(y, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(y)
    s = s * 5
    y = cv.merge((h, s, v))
    x = frame.copy()
    b, g, r = cv.split(x)
    b = cv.equalizeHist(b)
    g = cv.equalizeHist(g)
    r = cv.multiply(r, np.array([10]))  # type :ignore
    oia = cv.merge((b, g, r))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Video", frame)
    cv.imshow("Original Image", oia)
    cv.imshow("saturated", y)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
