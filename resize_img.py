import cv2


img = cv2.imread('pic/elephant.jpg')
cv2.imshow('test', img)

img2 = cv2.resize(img, 224,224)

cv2.waitKey(0)