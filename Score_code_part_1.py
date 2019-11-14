
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

print ("Directories")

base_dir = r'C:\reet_personal\hackathon\input\base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')

nv = os.path.join(train_dir, 'nv')
mel = os.path.join(train_dir, 'mel')
bkl = os.path.join(train_dir, 'bkl')
bcc = os.path.join(train_dir, 'bcc')
akiec = os.path.join(train_dir, 'akiec')
vasc = os.path.join(train_dir, 'vasc')
df = os.path.join(train_dir, 'df')

nv = os.path.join(val_dir, 'nv')
mel = os.path.join(val_dir, 'mel')
bkl = os.path.join(val_dir, 'bkl')
bcc = os.path.join(val_dir, 'bcc')
akiec = os.path.join(val_dir, 'akiec')
vasc = os.path.join(val_dir, 'vasc')
df = os.path.join(val_dir, 'df')

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

print ("Train and validation index created")

print ("Score_code_part_1_end, start part 2")