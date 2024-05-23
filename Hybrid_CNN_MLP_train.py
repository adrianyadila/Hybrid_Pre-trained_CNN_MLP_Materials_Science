import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras import regularizers, optimizers
from keras.optimizers import Adam
from keras.layers import Concatenate 
from keras.layers import Average
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import NumpyArrayIterator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import image 
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import cv2
import tensorflow as tf
import keras as k
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib 
import argparse
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import random
import pickle
import cv2
import os
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

ap = argparse.ArgumentParser()
ap.add_argument("-csv", "--traincsv", required=True,
	help="path to input dataset (i.e., directory of numerical data)")
ap.add_argument("-img", "--trainimgs", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model",  required=True,
	help="path to output model")
ap.add_argument("-hist", "--history", required=True,
	help="path to output model's history")
args = vars(ap.parse_args())

INIT_LR = 0.001
EPOCHS = 100
BS = 64

train_csv_path = args["traincsv"]
train_image_path = args["trainimgs"]

train_df= pd.read_csv(train_csv_path)
print(train_df.head())

train_X, validation_X, train_Y, validation_Y = train_test_split(train_df.drop(columns=["hardness1", "hardness2", "hardness3", "hardness4"] , axis="columns"), train_df[["hardness1", "hardness2", "hardness3", "hardness4"]], test_size=0.2, shuffle=True)
print(train_X.shape, validation_X.shape)
print(train_Y.shape, validation_Y.shape)

train_X = train_X.reset_index(drop =True)
train_X_mlp= train_X.drop(columns=["Id"], axis="columns")
validation_X_mlp= validation_X.drop(columns=["Id"], axis="columns")
print(validation_X_mlp.shape)
print(train_X_mlp.shape)

train_X_mlp = np.array(train_X_mlp)
train_Y = np.array(train_Y)

validation_X_mlp = np.array(validation_X_mlp)
validation_Y = np.array(validation_Y)

def get_image_array(train_df, validation_df):
    train_images = []
    validation_images = []
    train_image_path = args["trainimgs"] 

    for img_name in train_df['Id']:
        img_path = f"{train_image_path}\{img_name}"
        image = cv2.imread(img_path)
        image = cv2.resize(image, (90,120))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_images.append(image)
   
    for validation_img_name in validation_df['Id']:
        val_img_path = f"{train_image_path}\{validation_img_name}"
        val_image = cv2.imread(val_img_path)
        val_image = cv2.resize(val_image, (90, 120))
        val_image = cv2.cvtColor(val_image, cv2.COLOR_BGR2GRAY)
        validation_images.append(val_image)
       
    return np.array(train_images), np.array(validation_images)

train_images, validation_images = get_image_array(train_X, validation_X)
#print(train_images)
train_images = np.array(train_images, dtype="float") / 255.0
print(train_images.shape)
validation_images = np.array(validation_images, dtype="float") / 255.0

train_images = train_images.reshape(5113, 120, 90, 1)
print(train_images.shape)

validation_images = validation_images.reshape(1279, 120, 90, 1)
print(validation_images.shape)

#construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True, fill_mode="constant")

def aug_flow_for_two_inputs(X1, X2, y):
    augX1 = aug.flow(X1,y, batch_size=BS, shuffle=False)
    augX2 = aug.flow(X1,X2, batch_size=BS, shuffle=False)
    while True:
            X1i = augX1.next()
            X2i = augX2.next()
            yield [X2i[1], X1i[0]], X1i[1]

aug_flow = aug_flow_for_two_inputs(train_images, train_X_mlp, train_Y)

aug_val_flow = aug_flow_for_two_inputs(validation_images, validation_X_mlp, validation_Y)

print(train_images.shape)
print(train_X_mlp.shape)
print(train_Y.shape)

def create_mlp(dims):
    model = Sequential([
        Dense(40, input_dim=dims, activation="relu"),
        Dropout(0.2),
        Dense(40, activation="relu"),
        Dropout(0.2),
        Dense(40, input_dim=dims, activation="relu"),
        Dropout(0.2),
        Dense(40, activation="relu"),
        Dropout(0.2),
        Dense(40, input_dim=dims, activation="relu"),
        Dropout(0.2),
        Dense(40, activation="relu"),
        Dropout(0.2),
        Dense(4, activation="relu")
    ])
    return model

def create_cnn(dims):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,  kernel_size=(3,3), input_shape=dims, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64,  kernel_size=(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=64,  kernel_size=(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    return model

mlp = create_mlp(2)
cnn = create_cnn((120, 90, 1))
combinedInput = Average()([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)
x = Dense(4, activation="softmax")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)
model.summary()

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=20, min_lr=1e-6, verbose=1)

opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
opt1= tf.keras.optimizers.legacy.SGD(learning_rate=INIT_LR)
opt2= tf.keras.optimizers.legacy.RMSprop(learning_rate=INIT_LR)
opt3= tf.keras.optimizers.legacy.SGD(learning_rate=INIT_LR, momentum=0.9)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ["accuracy"])

H = model.fit(aug_flow, validation_data=aug_val_flow, epochs=EPOCHS, steps_per_epoch=len(train_images) // BS,validation_steps=len(validation_images) // BS, verbose=2, callbacks=[reduce_lr])

plot_model(model, to_file='hybrid_model.png', show_shapes=True, show_layer_names=True)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the history:
print("[INFO] serializing history...")
f = open(args["history"], 'wb')
pickle.dump(H.history, f)
f.close()

# plot the training loss and accuracy
plt.rcParams['backend'] = 'tkagg'

plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["accuracy"], label="train")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.show()

plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper left")
plt.show()