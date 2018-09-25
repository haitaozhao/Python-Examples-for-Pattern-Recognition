import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

MYDIR = r"C:\train"
MYTEST = r"C:\test"
IMG_SIZE  =32

# for img in os.listdir(MYDIR):  # iterate over each image per dogs and cats
#     print(os.path.join(MYDIR,img))
#     img_array = cv2.imread(os.path.join(MYDIR,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
# #     plt.imshow(img_array, cmap='gray')
# #     plt.show()
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     label = 1 if img[:3]=='cat' else 0

#     break

def creat_training_data(myDir):
    Training_data = []
    for img in os.listdir(myDir):
        try:
            img_array = cv2.imread(os.path.join(myDir,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            label = 0 if img[:3]=='cat' else 1
            Training_data.append([new_array,label])
        except Exception as e:
            print('error')
            pass
    return Training_data

def creat_testing_data(myDir):
    Testing_data = []
    for img in os.listdir(myDir):
        try:
            img_array = cv2.imread(os.path.join(myDir,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            Testing_data.append([new_array,int(img[:-4])])
        except Exception as e:
            print('error')
            pass
    return Testing_data

myTraining_data = creat_training_data(MYDIR)
myTesting_data = creat_testing_data(MYTEST)
random.shuffle(myTraining_data)

X = []
y = []
Test = []
Id = []

for features,label in myTraining_data:
    X.append(features)
    y.append(label)

for testfeas,idx in myTesting_data:
    Test.append(testfeas)
    Id.append(idx)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Test = np.array(Test).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("test.pickle","wb")
pickle.dump(Test, pickle_out)
pickle_out.close()

pickle_out = open("id.pickle","wb")
pickle.dump(Id, pickle_out)
pickle_out.close()
