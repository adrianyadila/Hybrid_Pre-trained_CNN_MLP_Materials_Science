# https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

import matplotlib
#matplotlib.use("Agg")
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.utils import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from class_multiout import multiout
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from random import uniform, randint
import tensorflow_addons as tfa
from math import pi
from keras.utils import plot_model

IMAGE_DIMS = (120, 90, 1)
EPOCHS = 300

# construct the argument parser and parse the arguments
# HELP = python D:/Users/mathe/ML/Microestruturas/Cassete_multilosses_tt.py -d D:/Users/mathe/ML/Microestruturas/DATA/data_multioutput_tocsv
# python D:/Users/mathe/ML/Microestruturas/Cassete_multilosses_tt.py -d D:/Users/mathe/ML/Microestruturas/DATA/data_multioutput_tocsv -m D:/Users/mathe/ML/Microestruturas/
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="D:/Users/mathe/ML/Microestruturas/DATA/data_multioutput_tocsv")
ap.add_argument("-m", "--model", required=True,
	help="D:/Users/mathe/ML/Microestruturas/MyModel.h5")
ap.add_argument("-hist", "--history", required=True,
	help="path to output model's history")
"""ap.add_argument("-l", "--categorybin", required=True,
	help="path to output category label binarizer")
ap.add_argument("-c", "--colorbin", required=True,
	help="path to output color label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output",
	help="base filename for generated plots")"""
args = vars(ap.parse_args())

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, material location labels along with the passes labels 
data = []
LocationLabels = []
PassLabels = []

seed, upper, lower = 42, 90 * (pi/180.0), 0 * (pi/180.0)
def random_degree():
        return uniform(lower, upper)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]), interpolation=cv2.INTER_NEAREST)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = img_to_array(image)
	image = tfa.image.rotate(image, random_degree(), fill_mode='constant', fill_value=0)
	image = tf.image.stateless_random_flip_left_right(image, (seed, 0))
	image = tf.image.stateless_random_brightness(image, 1.0, (seed, 0))
	data.append(image)
	# extract the material pass and location from the path and update the respective lists
	(loc, passes) = imagePath.split(os.path.sep)[-2].split("_")
	LocationLabels.append(loc)
	PassLabels.append(passes)
print(f'example: {LocationLabels[0]} and {PassLabels[0]}')

print(f'{LocationLabels[1], PassLabels[1]}')
plt.figure(figsize=(13, 13))
plt.imshow(data[1], cmap='gray')
plt.show()
print('-=-' * 20)

# scale the raw pixel intensities to the range [0, 1] and convert to a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))
# convert the label lists to NumPy arrays prior to binarization
LocationLabels = np.array(LocationLabels)
PassLabels = np.array(PassLabels)
# binarize both sets of labels
print("[INFO] binarizing labels...")
LocationLB = LabelBinarizer()
PassLB = LabelBinarizer()
LocationLabels = LocationLB.fit_transform(LocationLabels)
PassLabels = PassLB.fit_transform(PassLabels)

split2 = train_test_split(data, LocationLabels, PassLabels, test_size=0.0087, random_state=42)
(trainX, testX, trainLocationY, testLocationY, trainPassY, testPassY) = split2
# partition the data into training and testing splits using 70% of the data for training and the remaining 30% for testing
split = train_test_split(data, LocationLabels, PassLabels, test_size=0.30555554, random_state=42)
(trainX, valX, trainLocationY, valLocationY, trainPassY, valPassY) = split

print(f'\033[1;32mtranx and valx len: {len(trainX)}, {len(valX)}\033[m')
print(f'\033[1;33mtestx len: {len(testX)}\033[m')
print('-=-' * 20)

# initialize our Multiout multi-output network
model = multiout.build(inputShape=(120, 90, 1), numLocations=(len(LocationLB.classes_)-1), numPass=len(PassLB.classes_), finalActl='sigmoid', finalActp='linear')
# plot_model(model, "D:/Users/mathe/ML/Microestruturas/arcR.png", show_shapes=True)

# define two dictionaries: one that specifies the loss method for each output of the network along with a second dictionary that specifies the weight per loss
loc_loss = tf.keras.losses.BinaryCrossentropy()
pass_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
losses = {
	"location_output": loc_loss,
	"pass_output": pass_loss,
}
lossWeights = {"location_output": 0.429, "pass_output": 0.571}


# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = SGD(momentum=0.8) # 0.8
opt2 = Adam()
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

H = model.fit(x=trainX, y={"location_output": trainLocationY, "pass_output": trainPassY},
			validation_data=(valX, {"location_output": valLocationY, "pass_output": valPassY}),
	        epochs=EPOCHS, batch_size=200, validation_batch_size=44, verbose=2) # 200, 44 

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
#multioutputweights = model.save_weights('model_weights.h5')

# save the history:
print("[INFO] serializing history...")
f = open(args["history"], 'wb')
pickle.dump(H.history, f)
f.close()

"""# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open(args["categorybin"], "wb")
f.write(pickle.dumps(categoryLB))
f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()"""

print('-=-' * 20)

# plot the total loss, location loss, and pass loss
lossNames = ["loss", "location_output_loss", "pass_output_loss"]
plt.rcParams['backend'] = 'tkagg'
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
plt.show()
"""# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()"""

# create a new figure for the accuracies
accuracyNames = ["location_output_accuracy", "pass_output_accuracy"]
plt.rcParams['backend'] = 'tkagg'
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
plt.show()
"""# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()"""


