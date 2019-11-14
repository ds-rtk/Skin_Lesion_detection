
print ("Import Required libraries")
#####
import tensorflowjs as tfjs
from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
# %matplotlib inline



print ("Loading data")

path = r'C:\reet_personal\hackathon\input'
os.chdir(path)

print ("Directories")
######################
# base_dir = r'C:\reet_personal\hackathon\input\base_dir'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')

# nv = os.path.join(train_dir, 'nv')
# mel = os.path.join(train_dir, 'mel')
# bkl = os.path.join(train_dir, 'bkl')
# bcc = os.path.join(train_dir, 'bcc')
# akiec = os.path.join(train_dir, 'akiec')
# vasc = os.path.join(train_dir, 'vasc')
# df = os.path.join(train_dir, 'df')

# nv = os.path.join(val_dir, 'nv')
# mel = os.path.join(val_dir, 'mel')
# bkl = os.path.join(val_dir, 'bkl')
# bcc = os.path.join(val_dir, 'bcc')
# akiec = os.path.join(val_dir, 'akiec')
# vasc = os.path.join(val_dir, 'vasc')
# df = os.path.join(val_dir, 'df')
######################

print ("Reading metadata")

df_data = pd.read_csv(r'C:\reet_personal\hackathon\input\HAM10000_metadata.csv')
df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

print ("Identifying duplicates")

def dup(x):
    
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
df_data['duplicates'] = df_data['lesion_id']

# apply the function to this new column
df_data['duplicates'] = df_data['duplicates'].apply(dup)
df = df_data[df_data['duplicates'] == 'no_duplicates']

y = df['dx']
_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)


print ("validation rows")

def id_val_rows (x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'


df_data['train_or_val'] = df_data['image_id']
df_data['train_or_val'] = df_data['train_or_val'].apply(id_val_rows )
   
df_train = df_data[df_data['train_or_val'] == 'train']

df_train = df_train[:10]
df_val = df_val[:10]

print(len(df_train))
print(len(df_val))

print ("Train and validation index created")

print ("Score_code_part_1_end, start part 2")



print ("Score code part 2 start")


# Set the image_id as the index in df_data
df_data.set_index('image_id', inplace=True)

# Change the folder name
###################################################################################################################################


# # Change path
# folder_1 = os.listdir(r'C:\reet_personal\hackathon\input\ham10000_images_part_1')
# folder_2 = os.listdir(r'C:\reet_personal\hackathon\input\ham10000_images_part_2')

# train_list = list(df_train['image_id'])
# val_list = list(df_val['image_id'])

# print ("Transfer the train images")

# # Transfer the train images

# for image in train_list:
    
    # fname = image + '.jpg'
    # label = df_data.loc[image,'dx']
    
    # if fname in folder_1:
		# # Change path
        # src = os.path.join(r'C:\reet_personal\hackathon\input\ham10000_images_part_1', fname)
        # # destination path to image
        # dst = os.path.join(train_dir, label, fname)
        # # copy the image from the source to the destination
        # shutil.copyfile(src, dst)

    # if fname in folder_2:
		# # Change path
        # src = os.path.join(r'C:\reet_personal\hackathon\input\ham10000_images_part_2', fname)
        # # destination path to image
        # dst = os.path.join(train_dir, label, fname)
        # # copy the image from the source to the destination
        # shutil.copyfile(src, dst)


# print ("Transfer the val images")

# for image in val_list:
    
    # fname = image + '.jpg'
    # label = df_data.loc[image,'dx']
    
    # if fname in folder_1:
        # # Change path
        # src = os.path.join(r'C:\reet_personal\hackathon\input\ham10000_images_part_1', fname)
        # # destination path to image
        # dst = os.path.join(val_dir, label, fname)
        # # copy the image from the source to the destination
        # shutil.copyfile(src, dst)

    # if fname in folder_2:
        # # Change path
        # src = os.path.join(r'C:\reet_personal\hackathon\input\ham10000_images_part_2', fname)
        # # destination path to image
        # dst = os.path.join(val_dir, label, fname)
        # # copy the image from the source to the destination
        # shutil.copyfile(src, dst)
		

		
# # note that we are not augmenting class 'nv'
# class_list = ['mel','bkl','bcc','akiec','vasc','df']


# for item in class_list:
    
        # # create a base dir
    # aug_dir = 'aug_dir'
    # os.mkdir(aug_dir)
    # # create a dir within the base dir to store images of the same class
    # img_dir = os.path.join(aug_dir, 'img_dir')
    # os.mkdir(img_dir)

    # # Choose a class
    # img_class = item

    # # list all images in that directory
    # img_list = os.listdir('base_dir/train_dir/' + img_class)

    # # Copy images from the class train dir to the img_dir e.g. class 'mel'
    # for fname in img_list:
            # # source path to image
            # src = os.path.join('base_dir/train_dir/' + img_class, fname)
            # # destination path to image
            # dst = os.path.join(img_dir, fname)
            # # copy the image from the source to the destination
            # shutil.copyfile(src, dst)


    # # point to a dir containing the images and not to the images themselves
    # path = aug_dir
    # save_path = 'base_dir/train_dir/' + img_class

    # # Create a data generator
    # datagen = ImageDataGenerator(
        # rotation_range=180,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # zoom_range=0.1,
        # horizontal_flip=True,
        # vertical_flip=True,
        # #brightness_range=(0.9,1.1),
        # fill_mode='nearest')

    # batch_size = 50

    # aug_datagen = datagen.flow_from_directory(path,
                                           # save_to_dir=save_path,
                                           # save_format='jpg',
                                                    # target_size=(224,224),
                                                    # batch_size=batch_size)



    # # Generate the augmented images and add them to the training folders
    
    # ###########
    
    # num_aug_images_wanted = 6000 # total number of images we want to have in each class
    
    # ###########
    
    # num_files = len(os.listdir(img_dir))
    # num_batches = int(np.ceil((num_aug_images_wanted-num_files)/batch_size))

    # # run the generator and create about 6000 augmented images
    # for i in range(0,num_batches):

        # imgs, labels = next(aug_datagen)
        
    # # delete temporary directory with the raw image files
    # shutil.rmtree('aug_dir')
	
###################################################################################################################################################

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)



print ("Image generator starts")

datagen = ImageDataGenerator(
    preprocessing_function= \
    tensorflow.keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

# Note: shuffle=False causes the test dataset to not be shuffled
test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)
											

											
											
											
print ("Model mobilenet import")
											
mobile = tensorflow.keras.applications.mobilenet.MobileNet()

# CREATE THE MODEL ARCHITECTURE

# Exclude the last 5 layers of the above model.
# This will include all layers up to and including global_average_pooling2d_1
x = mobile.layers[-6].output

# Create a new dense layer for predictions
# 7 corresponds to the number of classes
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)


# inputs=mobile.input selects the input layer, outputs=predictions refers to the
# dense layer we created above.

model = Model(inputs=mobile.input, outputs=predictions)

# We need to choose how many layers we actually want to be trained.

# Here we are freezing the weights of all layers except the
# last 23 layers in the new model.
# The last 23 layers of the model will be trained.

for layer in model.layers[:-23]:
    layer.trainable = False


# Define Top2 and Top3 Accuracy

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)




model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])



# Get the labels that are associated with each index
print(valid_batches.class_indices)


filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
    5: 1.0, # nv
    6: 1.0, # vasc
}


history = model.fit_generator(train_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=1, verbose=1,
                   callbacks=callbacks_list)

# Here the the last epoch will be used.


tfjs.converters.save_keras_model(model, r"C:\reet_personal\hackathon\model_output") 
print ("Model output stored")

print ("Validation Results")


val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
model.evaluate_generator(test_batches, 
                        steps=len(df_val))

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)


											