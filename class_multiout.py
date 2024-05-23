import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Lambda
from keras.models import Model
import numpy as np
import tensorflow_addons as tfa
from random import uniform, randint
from math import pi
from keras.utils import to_categorical
from PIL import Image
import pandas as pd
import os
import glob
from keras.constraints import max_norm


class multiout:
        
	@staticmethod
	def build_location_branch(inputs, numlocations, finalAct="sigmoid"):
		M1_l1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='M1_layer_1')(inputs) # 16
		M1_l2 = MaxPooling2D((2, 2), padding='valid', name='M1_layer_2')(M1_l1)
		M1_l3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='M1_layer_3')(M1_l2) # 32
		M1_l4 = MaxPooling2D((2, 2), padding='same', name='M1_layer_4')(M1_l3)
		M1_l5 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='M1_layer_5')(M1_l4) # 64
		M1_l6 = MaxPooling2D((2, 2), padding='valid', name='M1_layer_6')(M1_l5)
		M1_l7 = Flatten(name='M1_layer_7')(M1_l6)
		M1_l8 = Dense(64, activation='relu', name='M1_layer_8')(M1_l7) # 32
		M1 = Dense(numlocations, activation=finalAct, name='location_output')(M1_l8)
		return M1

	@staticmethod
	def build_pass_branch(inputs, numPass, finalAct="softmax"):
		M2_l1 = Conv2D(32, (3, 3), padding='same', activation='relu', name='M2_layer_1', kernel_constraint=max_norm(4))(inputs) # 32
		M2_l2 = MaxPooling2D((2, 2), padding='valid', name='M2_layer_2')(M2_l1)
		M2_ld1 = Dropout(.25)(M2_l2)
		M2_l3 = Conv2D(64, (3, 3), padding='same', activation='relu', name='M2_layer_3', kernel_constraint=max_norm(4))(M2_ld1) # 64
		M2_l4 = MaxPooling2D((2, 2), padding='same', name='M2_layer_4')(M2_l3)
		M2_ld2 = Dropout(.2)(M2_l4)
		M2_l5 = Conv2D(128, (3, 3), padding='valid', activation='relu', name='M2_layer_5', kernel_constraint=max_norm(4))(M2_ld2) # 128
		M2_l6 = MaxPooling2D((2, 2), padding='valid', name='M2_layer_6')(M2_l5)
		M2_ld3 = Dropout(.25)(M2_l6)
		M2_l7 = Conv2D(256, (3, 3), padding='same', activation='relu', name='M2_layer_7', kernel_constraint=max_norm(4))(M2_ld3) # 256
		M2_l8 = MaxPooling2D((3, 3), padding='valid', name='M2_layer_8')(M2_l7)
		M2_ld4 = Dropout(.2)(M2_l8)
		M2_l9 = Flatten(name='M2_layer_9')(M2_ld4)
		M2_l10 = Dense(512, activation='relu', name='M2_layer_10')(M2_l9) # 512
		M2_ld5 = Dropout(.25)(M2_l10)
		M2 = Dense(numPass, activation=finalAct, name='pass_output')(M2_ld5)
		return M2
	
	def build(inputShape, numLocations, numPass, finalActl="sigmoid", finalActp='softmax'):
		#inputs = (width, height, 1)

		# construct both the "location" and "pass" sub-networks
		inputs = Input(shape=inputShape, name='input_start')
		locationBranch = multiout.build_location_branch(inputs,
			numLocations, finalAct=finalActl)
		passBranch = multiout.build_pass_branch(inputs,
			numPass, finalAct=finalActp)
		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the location category
		# branch and another for the passes branch, respectively
		model = Model(
			inputs=inputs,
			outputs=[locationBranch, passBranch],
			name="multiout_model")
		# return the constructed network architecture
		return model





dataset_folder_name = 'Microestruturas'
TRAIN_val_SPLIT = 0.6944444445
IM_WIDTH, IM_HEIGHT, CHANNELS = 160, 120, 1
seed, upper, lower = randint(1, 100), 90 * (pi/180.0), 0 * (pi/180.0)

dataset_dict = {
    'pass_id': {
        0: 'PASS1', 
        1: 'PASS2', 
        2: 'PASS3', 
        3: 'PASS4'
    },
    'location_id': {
        10: 'BORDER',
        11: 'CENTER'
    }
}
dataset_dict['location_alias'] = dict((g, i) for i, g in dataset_dict['location_id'].items())
dataset_dict['pass_alias'] = dict((r, i) for i, r in dataset_dict['pass_id'].items())

def random_degree():
        return uniform(lower, upper)

class prepro:
    def __init__(self, df):
        self.df = df

    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_val_SPLIT)
        train_idx = p[:train_up_to]
        valid_idx = p[train_up_to:]

        # converts alias to id
        self.df['location_id'] = self.df['location_label'].map(lambda location: dataset_dict['location_alias'][location])
        self.df['pass_id'] = self.df['passes_label'].map(lambda passes: dataset_dict['pass_alias'][passes])
        return train_idx, valid_idx

    def preprocess_image(self, folder_path):
        image_paths = glob.glob(folder_path + "/*.png")  # Altere a extensão do arquivo conforme necessário
        processed_images = []
        
        for img_path in image_paths:
            image_string = tf.io.read_file(img_path)
            image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
            image_resized = tf.image.resize(image_decoded, [IM_WIDTH, IM_HEIGHT])
            image_normalized = image_resized / 255.0

            image_brightness = tf.image.stateless_random_brightness(image_normalized, 0.5, (seed, 0))
            image_fliphor = tf.image.stateless_random_flip_left_right(image_brightness, (seed, 0))
            image_flipver = tf.image.stateless_random_flip_up_down(image_fliphor, (seed, 0))
            im = tfa.image.rotate(image_flipver, random_degree())
            
            processed_images.append(im)
        
        return processed_images

    def generate_images(self, image_idx, is_training, batch_size):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, locations, passes = [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]

                location = person['location_id']
                Pass = person['pass_id']
                file = person['file_path']
                
                processed_images = self.preprocess_image(file)
                
                for im in processed_images:
                    locations.append(to_categorical(location, len(dataset_dict['location_id'])))
                    passes.append(to_categorical(Pass, len(dataset_dict['pass_id'])))
                    images.append(im)
                    
                    # yielding condition
                    if len(images) >= batch_size:
                        yield np.array(images), [np.array(locations), np.array(passes)]
                        images, locations, passes = [], [], []
                        
            if not is_training:
                break


class prepro_data:
    def __init__(self, df):
        self.df = df
    
    def preprocess_image(self, img_path):
        image_string = tf.io.read_file(img_path)
        # Decode it into a dense vector
        image_decoded = tf.image.decode_png(image_string, channels=CHANNELS)
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, [IM_WIDTH, IM_HEIGHT])
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0

        image_brightness = tf.image.stateless_random_brightness(image_normalized, 0.5, (seed, 0))
        image_fliphor = tf.image.stateless_random_flip_left_right(image_brightness, (seed, 0))
        image_flipver = tf.image.stateless_random_flip_up_down(image_fliphor, (seed, 0))
        im = tfa.image.rotate(image_flipver, random_degree())
        return im
    