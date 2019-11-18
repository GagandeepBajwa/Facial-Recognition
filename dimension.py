import cv2

impath = "/Users/BradleyCrump/Documents/school/fall/c4980/project/lfw-dataset/lfw-deepfunneled/Frank_Marshall/Frank_Marshall_0001.jpg"

im = cv2.imread(impath, cv2.IMREAD_UNCHANGED)

print(im.shape)
