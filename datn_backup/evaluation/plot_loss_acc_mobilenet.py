
import pandas as pd
import pylab as plt
import os
fileDir = os.path.dirname(os.path.realpath(__file__)) 
file_name = 'training_mobilenetv2ssd512_class_weight_last.log'
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