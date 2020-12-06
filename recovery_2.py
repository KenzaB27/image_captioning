# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import matplotlib.pyplot as plt
    
import numpy as np
import pandas as pd
import keras
import os
import pickle
import random
from tqdm import tqdm
from IPython.display import Image, display

from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence


# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# %%
INPUT_IMG_SIZE = (224, 224, 3)
CNN_OUTPUT_SIZE = 2048

START_TOKEN = '<start>'
END_TOKEN = '<end>'

BATCH_IMAGES = 2500

EMBEDDING_SIZE = 128

# %% [markdown]
# # Preprocessing
# ## Load Data

# %%
# adding a start and end token
def add_start_end_tokens(captions):
   new_captions = ['']*len(captions)
   for i in range(len(captions)):
      new_captions[i] = '<start> ' + captions[i] + ' <end>'
   return new_captions


# %%
captions_path = "captions.txt"
images_path = "Images/"


# %%
captions_df = pd.read_csv(captions_path, delimiter = ',')
captions_df.head(5)


# %%
captions_df.shape


# %%
captions_to_list_df = captions_df.groupby('image')['caption'].apply(list).reset_index(name='captions')
captions_to_list_df['captions'] = captions_to_list_df['captions'].apply(add_start_end_tokens)
captions_to_list_df.head(5)


# %%
captions_to_list_df.shape


# %%
img = Image(filename=images_path+captions_to_list_df['image'][0])
display(img)
print(captions_to_list_df['captions'][0])

# %% [markdown]
# # CNN Model for image processing

# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(captions_to_list_df, test_size=0.2)


# %%
train_set.shape


# %%
# unfold the train data set 
lst_col = 'captions'

train_set_unfolded = pd.DataFrame({
      col:np.repeat(train_set[col].values, train_set[lst_col].str.len())
      for col in train_set.columns.drop(lst_col)}
    ).assign(**{lst_col:np.concatenate(train_set[lst_col].values)})[train_set.columns]
train_set_unfolded.head(5)


# %%
train_set_unfolded.shape


# %%
# include_top: whether to include the fully_connected layer at the end of the network
# weights: image_net (pre-training on ImageNet)
# pooling: Optional pooling moe for feature extraction when include_top is False
cnn_model = ResNet50(include_top=False, weights='imagenet', input_shape=INPUT_IMG_SIZE, pooling='avg')


# %%
cnn_model.summary()

# %% [markdown]
# ## Preprocessing the images for predicting their labels with pretrained model

# %%
def image_preprocessing(img_path):
   im = image.load_img(img_path, target_size=INPUT_IMG_SIZE)
   im = image.img_to_array(im)
   im = np.expand_dims(im, axis=0)
   return im


# %%
train_data_im = {}
for img_path in tqdm(train_set['image']):
   path = images_path + img_path
   img = image_preprocessing(path)
   pred = cnn_model.predict(img).reshape(CNN_OUTPUT_SIZE)
   train_data_im[img_path] = pred


# %%
train_data_im['667626_18933d713e.jpg']

# %% [markdown]
# ## Preprocessing the descriptions 
# ### Extracting the sentences

# %%
sentences = [s for cap in captions_to_list_df['captions'] for s in cap]    
print(len(sentences))


# %%
max_len = 0

for i in sentences:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)

print(max_len)

# %% [markdown]
# ### Extracting unique words

# %%
words = [s.split() for s in captions_df['caption']] + [[START_TOKEN, END_TOKEN]]
words[-1]


# %%
unique = []

for word in words:
   unique.extend(word)

unique = list(set(unique))

vocab_size = len(unique)
vocab_size


# %%
# Vectorization
word_2_indices = {val:index for index, val in enumerate(unique)}
indices_2_word = {index:val for index, val in enumerate(unique)}


# %%
word_2_indices['UNK'] = vocab_size
indices_2_word[vocab_size] = 'UNK'


# %%
print(word_2_indices['<start>'])
print(indices_2_word[4011])
print(word_2_indices['<end>'])
print(indices_2_word[8051])


# %%
vocab_size = len(word_2_indices.keys())
print(vocab_size)

# %% [markdown]
# ### Encoding the sentences and padding them
# We need to have the sentences as sequences so that we can feed the LSTM neurons.
# 
# The subsequen_words are the labels of the LSTM: the prediction of the next word.

# %%
padded_sequences, subsequent_words = [], []

for ix in tqdm(range(len(sentences))):
    partial_seqs = []
    next_words = []
    text = sentences[ix].split()
    text = [word_2_indices[i] for i in text]
    # we start from index 1 to get the next word in the sentence
    for i in range(1, len(text)):
        partial_seqs.append(text[:i])
        next_words.append(text[i])
    padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_len, padding='post')

    next_words_1hot = np.zeros([len(next_words), vocab_size], dtype=np.bool)
    
    #Vectorization
    for i,next_word in enumerate(next_words):
        next_words_1hot[i, next_word] = 1
        
    padded_sequences.append(padded_partial_seqs)
    subsequent_words.append(next_words_1hot)
    
padded_sequences = np.asarray(padded_sequences)
subsequent_words = np.asarray(subsequent_words)

print(padded_sequences.shape)
print(subsequent_words.shape)


# %%
print(padded_sequences[0])


# %%
for i in range(len(padded_sequences[0])):
    for j in range(max_len):
        print(indices_2_word[padded_sequences[0][i][j]],end=" ")
    print("\n")

print(len(padded_sequences[0]))


# %%
# Example of how the labels work, prediction of the next word
for i in range(len(subsequent_words[0][1])):
    if subsequent_words[0][1][i]:
        print(i)


# %%
#save the captions and nex_words for training 
captions = np.zeros([0, max_len])
next_words = np.zeros([0, vocab_size])

for i in tqdm(range(BATCH_IMAGES)):#img_to_padded_seqs.shape[0]):
    captions = np.concatenate([captions, padded_sequences[i]])
    next_words = np.concatenate([next_words, subsequent_words[i]])

print(captions.shape)
print(next_words.shape)


# %%
imgs = []
ds = train_set_unfolded.values
for ix in range(ds.shape[0]):
    if ds[ix, 0] in train_data_im.keys():
#         print(ix, encoded_images[ds[ix, 0].encode()])
        imgs.append(train_data_im[ds[ix, 0]])

imgs = np.asarray(imgs)
print(imgs.shape)


# %%
# create an array of embeddings of the padding sequences
image_embeddings = []

for i in range(BATCH_IMAGES):
    for _ in range(padded_sequences[i].shape[0]):
        image_embeddings.append(imgs[i])
        
image_embeddings = np.asarray(image_embeddings)

print(image_embeddings.shape)


# %%
# create an array of image names of the padding sequences

image_names = []

for i in range(BATCH_IMAGES):
    for _ in range(padded_sequences[i].shape[0]):
        image_names.append(ds[i,0])
        
image_names = np.asarray(image_names)

print(image_names.shape)

# %% [markdown]
# # Image captioning model

# %%
# Checking the shapes of all the inputs
print(captions.shape)
print(next_words.shape)

print(image_embeddings.shape)
print(image_names.shape)


# %%
image_model = Sequential()

image_model.add(Dense(EMBEDDING_SIZE, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

image_model.summary()


# %%
language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

language_model.summary()


# %%
conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

# model.load_weights("../input/model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()


