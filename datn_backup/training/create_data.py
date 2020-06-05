# import os
# fileDir = os.path.dirname(os.path.realpath(__file__))
# testDir = os.path.join(fileDir, 'test.csv')
# trainDir = os.path.join(fileDir, 'train.csv')
# valDir = os.path.join(fileDir, 'val.csv')

# with open(testDir, 'r') as t1, open(trainDir, 'r') as t2, open(valDir, 'r') as t3:
#     test = t1.readlines()
#     train = t2.readlines()
#     val = t3.readlines()

# with open('train_new.csv', 'w') as outFile:
#     outFile.write(train[0])
#     for i in range(1, len(train)): 
#         arr = train[i].split(',')
#         arr[-1] = str(int(arr[-1]) + 1) 
#         line = ','.join(arr)
#         line += '\n'
#         outFile.write(line)

# with open('val_new.csv', 'w') as outFile:
#     outFile.write(val[0])
#     for i in range(1, len(val)): 
#         arr = val[i].split(',')
#         arr[-1] = str(int(arr[-1]) + 1) 
#         line = ','.join(arr)
#         line += '\n'
#         outFile.write(line)

# with open('test_new.csv', 'w') as outFile:
#     outFile.write(test[0])
#     for i in range(1, len(test)): 
#         arr = test[i].split(',')
#         arr[-1] = str(int(arr[-1]) + 1) 
#         line = ','.join(arr)
#         line += '\n'
#         outFile.write(line)

import numpy as np 
print(np.random.choice(7))