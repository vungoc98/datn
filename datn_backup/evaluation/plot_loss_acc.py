
import pandas as pd
import pylab as plt
import os
fileDir = os.path.dirname(os.path.realpath(__file__))
# file_name = 'training_vgg16ssd512_last (1).log'
# file_name = 'training_mobilenetv2ssd512.log'
# file_name = 'training_vgg16ssd300_last.log'
# file_name = 'training_vgg16ssd512_last.log'
# file_name = 'training_mobilenetv2ssd512_last_odd_scales_last.log'
# file_name = 'training_mobilenetv2ssd512.log'
# file_name = 'training_mobilenetv2ssd512_odd_scales_splitting_image.log'
# file_name = 'training_mobilenetv2ssd512_new_scales_splitting_image.log'
file_name = 'training_mobilenetv2ssd512_scales_for_splitting_image.log'
file_name = 'new.log'
file_name = os.path.join(fileDir, file_name)
df = pd.read_csv(file_name) 
 
lines = df.plot.line(x='epoch', y=['acc', 'val_acc']) 
plt.title('CNN learning curves')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='lower right')  

lines = df.plot.line(x='epoch', y=['loss', 'val_loss'])
plt.title('CNN learning curves')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper right') 
plt.show() 