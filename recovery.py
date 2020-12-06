# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import matplotlib.pyplot as plt
    
import numpy as np
import pandas as pd
import keras
import os
import random
from tqdm import tqdm
from IPython.display import Image, display

from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence


# %%
INPUT_IMG_SIZE = (224, 224, 3)
CNN_OUTPUT_SIZE = 2048

START_TOKEN = '<start>'
END_TOKEN = '<end>'

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
captions_to_list_df = captions_df.groupby('image')['caption'].apply(list).reset_index(name='captions')
captions_to_list_df['captions'] = captions_to_list_df['captions'].apply(add_start_end_tokens)
captions_to_list_df.head(5)


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



