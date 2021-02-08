# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
from LeNet import LeNet
from keras.models import load_model
import shutil
import pandas as pd 
def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
        help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
        help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args

EPOCHS = 300
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 2
norm_size = 32

t1,t0=[],[]
p1,p0=[],[]

def load_data(path):
    print("[INFO] loading images...")
    # print(path)
    data = []
    labels = []
    names = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # print(imagePaths)
    # loop over the input images
    global t1,t0
    for imagePath in imagePaths:
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list

        label = int(imagePath.split(os.path.sep)[-2])       
        # print(label)
        # print(imagePath,label)
        if(label==1):
            t1.append(imagePath)
        else:
            t0.append(imagePath)

        labels.append(label)
        names.append(imagePath)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    l2=labels
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)                         
    return data,labels,names,l2

def train(aug,trainX,trainY,testX,testY,args):
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    

    # save the model to disk
    print("[INFO] serializing network...")
    model.save("model2.h5")
    # model = load_model("model.h5")
    # prediction=model.predict(testX)
    # print(prediction)
    # print(testY)
    # model.save("model.h5")

def predict(testX,testY,names,labels):
    model=load_model("model2.h5")
    model.summary()
    prediction=model.predict(testX)

    prediction=prediction.tolist()
    testY=testY.tolist()
    
    name=[]
    ans=[]
    global p1,p0
    for a,b,n in zip(prediction,testY,names):
        # print(n[6:])
        name.append(n[6:])
        ans.append(a[1])
        # print(b)
        # print(a)
        # print("----------") 

        if a[0]>a[1]:
            # if(label==1):
                # t1+=1
            # else
            # p0.append(n)
            shutil.move(n,"./Me/0")
        else:
            # if(label==1):
            # p1.append(n)
            # else
                # t0+=1
            shutil.move(n,"./Me/1")
    pd.DataFrame({"name":name,"sim":ans}).to_csv("CVSim.csv")
    # print(prediction)
    # print(testY)


    # model.save("model.h5")

#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    args = args_parse()
 
    # train_file_path = args["dataset_train"]
    test_file_path = args["dataset_test"]
 
    # trainX,trainY,x,y = load_data(train_file_path)
    testX,testY,names,l2 = load_data(test_file_path)
    predict(testX,testY,names,l2)

    # print([p0,p1],[t0,t1])
    # temp=0
    # for i in p0:
    #     if i in t0:
    #         temp+=1
    # print(temp)

    # temp=0
    # for i in p1:
    #     if i in t0:
    #         temp+=1
    # print(temp)

    # temp=0
    # for i in p0:
    #     if i in t1:
    #         temp+=1
    # print(temp)

    # temp=0
    # for i in p1:
    #     if i in t1:
    #         temp+=1
    # print(temp)


    # construct the image generator for data augmentation
    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #     horizontal_flip=True, fill_mode="nearest")

    # train(aug,trainX,trainY,testX,testY,args)