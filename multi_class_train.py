# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from multilabelcode.multinetwork import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
#ap.add_argument("-b", "--basemultioutput", required=True, help="path to basemultioutput")
ap.add_argument("-m", "--model",  required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-hist", "--history", required=True,
	help="path to output model's history")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",
	#help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#Get the dictionary of config for vgg16
#print("[INFO] loading multioutput-network...")
#base_multioutput = load_model(args["basemultioutput"])
#print(base_multioutput.summary())
#print(base_multioutput.get_config())

# Get the weights from the multioutput
#weights_multioutput = base_multioutput.get_weights()
#print('weights da basemultioutput')
#print(weights_multioutput)
#print(len(weights_multioutput))

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 0.001
BS = 64
IMAGE_DIMS = (120, 90, 1)
# disable eager execution
tf.compat.v1.disable_eager_execution()

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = img_to_array(image)
	data.append(image)
	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
	
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
	
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

print(trainX.shape)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="constant")

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=15, min_lr=1e-6, verbose=1)

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], classes=len(mlb.classes_),
	finalAct="sigmoid")
# initialize the optimizer (SGD is sufficient)
opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
opt1= tf.keras.optimizers.legacy.SGD(momentum=0.8)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print(model.summary())

# train the network
print("[INFO] training network...")
H = model.fit(x=trainX, y=trainY, batch_size=BS, 
	validation_data=(testX, testY), 
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=2, callbacks=[reduce_lr])

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the history:
print("[INFO] serializing history...")
f = open(args["history"], 'wb')
pickle.dump(H.history, f)
f.close()

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
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


