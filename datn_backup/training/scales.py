import os
import cv2
ratios = []
ratios1 = []
fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.join(fileDir, '..')
# image_dir = '/home/ubuntu/Documents/datn/ssd_keras/datasets/full' 
# gt_dir = '/home/ubuntu/Documents/datn/ssd_keras/gt.csv'
def get_ratio(image_dir, gt_dir): 
    global ratios
    with open(gt_dir, 'r') as gt:
        data = gt.readlines() 
    for d in data: 
        image_name, xmin, ymin, xmax, ymax, class_id = d.split(',')
        img = os.path.join(image_dir, image_name)
        img = cv2.imread(img)
        H, W, _ = img.shape
        h = float(ymax) - float(ymin)
        w = float(xmax) - float(xmin)
        # r = round(h / w, 2) 
        ratios.append(h / H)
        ratios.append(w / W)

train_dir = os.path.join(rootDir, 'train_full')
gt_train_dir = os.path.join(fileDir, 'train.csv')
get_ratio(train_dir, gt_train_dir)

val_dir = os.path.join(rootDir, 'val_full')
gt_val_dir = os.path.join(fileDir, 'val.csv')
get_ratio(val_dir, gt_val_dir)

test_dir = os.path.join(rootDir, 'test_full')
gt_test_dir = os.path.join(fileDir, 'test.csv')
get_ratio(test_dir, gt_test_dir)

# print(ratios)
print('min ratio: ', min(ratios))
print('max ratio: ', max(ratios))
# print('min ratio: ', min(ratios1))
# print('max ratio: ', max(ratios1)) 