import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')


import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from keras.models import Sequential, Model
from  keras.layers import Dense, Conv2D, Dropout, Flatten ,MaxPooling2D ,Input

BASE_DIR = 'C:\\Users\\User\\Downloads\\archive\\UTKFace'

import os
from tqdm import tqdm

image_paths = []
age_labels = []
gender_labels = []


for filename in tqdm(os.listdir(BASE_DIR)):
    
    
      
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

    
    



df = pd.DataFrame()
df['image'],df['age'],df['gender'] = image_paths,age_labels,gender_labels
df.head()

from PIL import Image

gender_dict = {0:'Male', 1:'Female'}

def extract_features(images):
    features=[]
    for image in tqdm(images):
        img=load_img(image, grayscale=True)
        img=img.resize((128,128),Image.ANTIALIAS)
        img=np.array(img)
        features.append(img)
    

    features = np.array(features)
    features=features.reshape(len(features),128,128,1)
    return features

x = extract_features(df['image'])
x.shape
# normalize the image
x=x/255.0
y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

input_shape = (128,128,1)


#model creation
inputs = Input((input_shape))
conv_1 = Conv2D(32, kernel_size=(3,3), activation='relu') (inputs)
maxp_1 =MaxPooling2D(pool_size=(2,2))(conv_1)

conv_2 = Conv2D(64, kernel_size=(3,3), activation='relu') (maxp_1)
maxp_2 =MaxPooling2D(pool_size=(2,2))(conv_2)

conv_3 = Conv2D(128, kernel_size=(3,3), activation='relu') (maxp_2)
maxp_3 =MaxPooling2D(pool_size=(2,2))(conv_3)

conv_4 = Conv2D(256, kernel_size=(3,3), activation='relu') (maxp_3)
maxp_4 =MaxPooling2D(pool_size=(2,2))(conv_4)

flatten = Flatten() (maxp_4)

#fully connected layers

dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy','mae'], optimizer='adam', metrics=['accuracy'])

history = model.fit(x=x, y=[y_gender,y_age], batch_size=32, epochs=10, validation_split=0.2)

model.save('trained_model.m5')

image_index = 234
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(x[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(x[image_index].reshape(128, 128), cmap='gray');