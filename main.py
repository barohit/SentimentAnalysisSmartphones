import os
import nltk
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense


current_directory = os.getcwd()

all_words = set()
data = []

#iterate through the list and extract the data and the labels
for filename in os.listdir(current_directory):
    label_num = -1
    if filename.endswith(".txt"):
        with open(filename, 'r') as file:
            text = file.read()
            if "iphone" in filename:
                label_num = 0
            elif "galaxy" in filename:
                label_num = 1
            elif "pixel" in filename:
                label_num = 2
            else:
                print(f"Invalid label found in {filename}")
                continue
            words = word_tokenize(text)
            data.append(words, label_num)
            all_words.update(words)


model = Sequential()

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
