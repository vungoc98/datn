### import library 
import os
from collections import defaultdict
import cv2
import shutil
  
def checkBB(gt, tile, size, stride):
    xmin, ymin, xmax, ymax, class_id = gt
    h, w, _ = size
    x, y, c_y, c_x = tile 

    # tinh toa do goc cua tile
    x_o = (c_x - 1) * stride
    y_o = (c_y - 1) * stride
    w_o = x_o + w
    h_o = y_o + h
    if x_o <= xmin and xmax <= w_o and y_o <= ymin and ymax <= h_o:
        return True
    return False

def findBB(gt, tile, stride):
    xmin, ymin, xmax, ymax, class_id = gt
    x, y, c_y, c_x= tile

    # tinh lai toa do bounding box tren tile
    xmin = xmin - (c_x - 1) * stride
    xmax = xmax - (c_x - 1) * stride
    ymin = ymin - (c_y - 1) * stride
    ymax = ymax - (c_y - 1) * stride
    return [xmin, ymin, xmax, ymax, class_id]

def split_image(image_name, labels):
    image = os.path.join(image_dir, image_name)
    image = cv2.imread(image)
    ## bo sung splitting     
    H, W, _ = image.shape
    stride = 400
    size = 512
    imgs = []
    input_size = []
    tiles = []
    count_x = 0
    count_y = 0 

    # splitting images into tiles with size and stride
    for y in range(0, H, stride):
        count_y += 1
        count_x = 0 
        for x in range(0, W, stride): 
            count_x += 1 
            img = image[y: min(y + size, H), x: min(x + size, W)]   
            imgs.append(img)
            input_size.append((x, y, count_y, count_x)) 
    
    # save gt 
    for gt in labels:
        xmin, ymin, xmax, ymax, class_id = gt
        gt = ','.join(gt)
        gt = image_name + ',' + gt + '\n'
        with open(train_output_dir, 'a') as of:
            of.write(gt)
    
    cv2.imwrite(os.path.join(output_image_dir, image_name), image)

    # chay tren tung tile de tim bounding box
    with open(train_output_dir, 'a') as of:
        for k in range(len(imgs)):
            tile_bbs = []
            x, y, c_y, c_x = input_size[k]
            c_y -= 1
            c_x -= 1 
            for gt in labels:  
                gt = list(map(int, gt)) 
                if checkBB(gt, input_size[k], imgs[k].shape, stride):
                    bb = findBB(gt, input_size[k], stride)
                    bb = list(map(str, bb))
                    # save into new file csv
                    bb = ','.join(bb)
                    n, ext = image_name.split('.')
                    img_name = n + str(100 + k) + '.ppm'
                    bb = img_name + ',' + bb + '\n' 
                    of.write(bb) 
                    cv2.imwrite(os.path.join(output_image_dir, img_name), imgs[k])


        
    ## done splitting image

fileDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.join(fileDir, '..')

image_dir = '/home/ubuntu/Documents/datn/ssd_keras/datasets/full'

### read from train.csv file
### split image and save in train.csv file
train_input_filename = 'test_new.csv'
train_output_filename = 'test.csv'
train_input_dir = os.path.join(fileDir, train_input_filename)
train_output_dir = os.path.join(fileDir, train_output_filename)
output_image_dir = os.path.join(rootDir, 'test_full')
if not os.path.exists(output_image_dir):
    os.mkdir(output_image_dir)
else:
    shutil.rmtree(output_image_dir)
    os.mkdir(output_image_dir)

with open(train_input_dir, 'r') as f:
    data = f.readlines()

images_dict = defaultdict(list) 
data = data[1:] # bo header
for d in data:
    d = d[:-1] # delete '\n'
    image_name, xmin, ymin, xmax, ymax, class_id = d.split(',') 
    images_dict[image_name].append([xmin, ymin, xmax, ymax, class_id])

for image_name, labels in images_dict.items():
    split_image(image_name, labels)