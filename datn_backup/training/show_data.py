import os
import cv2
fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.join(fileDir, '..')
image_dir = os.path.join(rootDir, 'test_full')

gt = os.path.join(fileDir, 'test.csv')

with open(gt, 'r') as f:
    data = f.readlines()

for d in data:
    d = d[:-1]
    image_name, xmin, ymin, xmax, ymax, class_id = d.split(',')
    image = cv2.imread(os.path.join(image_dir, image_name))
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    cv2.rectangle(image,(xmin, ymin), (xmax, ymax),(0,255,255),2)
    cv2.putText(image, class_id, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0),2) 
    cv2.imshow("image",image)
    cv2.waitKey(0)