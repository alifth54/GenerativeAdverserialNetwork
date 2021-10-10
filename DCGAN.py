#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:17:32 2021

@author: hp-pc
"""
import numpy as np
import keras as ks
import keras.layers as ksl
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam


def Bgen(zDim):
    gen = ks.models.Sequential()
    gen.add(ksl.Dense(256*7*7, input_dim=zDim))
    gen.add(ksl.Reshape([7, 7, 256]))

    gen.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    gen.add(ksl.BatchNormalization())
    gen.add(ksl.LeakyReLU(alpha=0.01))
    gen.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    gen.add(ksl.BatchNormalization())
    gen.add(ksl.LeakyReLU(alpha=0.01))
    gen.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    gen.add(ksl.Activation('tanh'))
    return gen


def Bdis(imShape):
    dis = ks.models.Sequential()
    dis.add(Conv2D(32, kernel_size=3, strides=2,
            input_shape=imShape, padding='same'))
    dis.add(ksl.LeakyReLU(alpha=0.01))
    dis.add(Conv2D(64, kernel_size=3, strides=2,
            input_shape=imShape, padding='same'))
    dis.add(ksl.BatchNormalization())
    dis.add(ksl.LeakyReLU(alpha=0.01))
    dis.add(Conv2D(128, kernel_size=3, strides=2,
            input_shape=imShape, padding='same'))
    dis.add(ksl.BatchNormalization())
    dis.add(ksl.LeakyReLU(alpha=0.01))
    dis.add(ksl.Flatten())
    dis.add(ksl.Dense(1, activation='sigmoid'))
    return dis

def Bmodel(dis,gen):
    model = ks.models.Sequential()
    model.add(gen)
    model.add(dis)
    return model

def Train(iteration,Batch):
    (xTrain,_),(_,_) = mnist.load_data()
    xTrain = np.expand_dims(xTrain,axis=3)
    xTrain = xTrain/127.5 -1.0
    real = np.ones([Batch,1])
    fake = np.zeros([Batch,1])
    for i in range(iteration):
        idx = np.random.randint(0,xTrain.shape[0],Batch)
        img = xTrain[idx]
        
        z=np.random.normal(0,1,[Batch,100])
        genIm = gen.predict(z)
        dis.train_on_batch(img,real)
        dis.train_on_batch(genIm,fake)
        z=np.random.normal(0,1,[Batch,100])
        model.train_on_batch(z,real)
        if i%100 == 0:
            print(i)
def show_images(gen):
    z = np.random.normal(0, 1, (16, 100))
    gen_imgs = gen.predict(z)
    gen_imgs = 0.5*gen_imgs + 0.5

    fig,axs = plt.subplots(4,4,figsize=(4,4),sharey=True,sharex=True)

    cnt=0
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(gen_imgs[cnt,:,:,0],cmap='gray')
            axs[i, j].axis('off')
            cnt+=1

    fig.show()
gen = Bgen(100)
dis = Bdis([28,28,1])
dis.compile(loss='mse',optimizer=Adam(),metrics=['accuracy'])
dis.trainable = False
model = Bmodel(dis,gen)

model.compile(loss='mse',optimizer=Adam())

Train(50,128)
show_images(gen)
    
    