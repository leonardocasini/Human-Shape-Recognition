import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from scipy.io import loadmat
import tqdm

import numpy as np
import os
import shutil
import cv2



import tensorflow.keras as K
from keras.optimizers import Adam




cwd = os.getcwd()

directoryRun = "cmu/train/"
directoryRun = os.path.join(cwd,directoryRun )


print(directoryRun)
images = []
thetas = []
count= 0
print("loading")
for folder in os.listdir(directoryRun):
    count = count +1
    print(count)
    directorySubFolderRun = os.path.join(directoryRun, folder)
    for filename in os.listdir(directorySubFolderRun):
        if filename.endswith("info.mat"):
            pathToInfoMat = os.path.join(directorySubFolderRun, filename)
            mat = loadmat(pathToInfoMat)
            seq = mat['sequence'][0]
            mp4filename = seq+".mp4"
            pathToVideo = os.path.join(directorySubFolderRun, mp4filename)
            theta = []
            for i in range(len(mat['shape'])):
                theta.append(mat['shape'][i][0])
            j = 0
            cap = cv2.VideoCapture(pathToVideo)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                    break
                if j%45 == 0:  #  == 0
                    image_normalized = (frame.astype(np.float32) - 127.5)/127.5 
                    images.append(image_normalized)
                    thetas.append(theta)
                j = j+1
            continue
        else:
            continue
    
        
print("Saving...")
#tf1 = tf.convert_to_tensor(images, dtype=tf.float32)
#tf2 = tf.convert_to_tensor(thetas, dtype=tf.float32)


images = np.array(images)
thetas = np.array(thetas)

print("loadin done")


GENERATOR_LEARNING_RATE = 1e-5

to_res = (224,224)

resnet50V2 = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg')
vs_initializer = tf.keras.initializers.VarianceScaling(2.0)

l2_regularizer = tf.keras.regularizers.l2(0.0010000000474974513)#non so se il valore è corretto

for layer in resnet50V2.layers:
            if isinstance(layer, layers.Conv2D):
                # original implementations slim `resnet_arg_scope` additionally sets
                # `normalizer_fn` and `normalizer_params` which in TF 2.0 need to be implemented
                # as own layers. This is not possible using keras ResNet50V2 application.
                # Nevertheless this is not needed as training seems to be likely stable.
                # See https://www.tensorflow.org/guide/migrate#a_note_on_slim_contriblayers for more
                # migration insights
                setattr(layer, 'padding', 'same')
                setattr(layer, 'kernel_initializer', vs_initializer)
                setattr(layer, 'kernel_regularizer', l2_regularizer)
            if isinstance(layer, layers.BatchNormalization):
                setattr(layer, 'momentum', 0.997)
                setattr(layer, 'epsilon', 1e-5)
            if isinstance(layer, layers.MaxPooling2D):
                setattr(layer, 'padding', 'same')
fc_one = layers.Dense(1024, name='fc_0',activation="relu")
dropout_one = layers.Dropout(0.5)
fc_two = layers.Dense(1024, name='fc_1',activation="relu")
dropout_two = layers.Dropout(0.5)
variance_scaling = tf.initializers.VarianceScaling(.01, mode='fan_avg', distribution='uniform')
fc_out = layers.Dense(10, kernel_initializer=variance_scaling, name='fc_out')

model = K.models.Sequential()
print("aleee")
model.add(K.layers.Lambda(lambda image: tf.image.resize(image,to_res)))
model.add(resnet50V2)
model.add(fc_one)
model.add(dropout_one)
model.add(fc_two)
model.add(dropout_two)
model.add(fc_out)

model.compile(loss="mean_squared_error",
              optimizer=Adam(GENERATOR_LEARNING_RATE)
             )
print("ciso")
model.fit(images,thetas,batch_size=64,epochs=1)
model.save(cwd+"/mymodel3")