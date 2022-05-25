import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

with open('F:/pycache/chatbot/intents.json') as file:
    data = json.load(file)

train_sentences = [ ]
train_labels =[ ] 
labels = [ ]
responses = [ ] 


for intent in data ['intents']:                     #load sentences from the json file into train_sentences and train_labels
    for pattern in intent['patterns']:
        train_sentences.append(pattern)
        train_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num = len(labels)        

lbl_encoder = LabelEncoder ( )
lbl_encoder.fit(train_labels)
train_labels = lbl_encoder.transform(train_labels)


vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<00V>"

tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
sequences =tokenizer.texts_to_sequences(train_sentences)        #converting text to sequence of charecters
padded_sequences = pad_sequences(sequences,truncating = 'post',maxlen = max_len)    #removing the spaces between the characters

###model training##

model =Sequential( )
model.add(Embedding(vocab_size, embedding_dim,input_length = max_len))
model.add(GlobalAveragePooling1D( ))
model.add(Dense(16,activation ='relu'))
model.add(Dense(16,activation ='relu'))
model.add(Dense(num,activation = 'softmax'))    #classifying the input based on the labels 

model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'adam',metrics = ['accuracy'])

#model.summary()

##fitiing the model###
epochs = 500
history = model.fit(padded_sequences , np.array(train_labels),epochs = epochs)

#save the trained model##
model.save("chat_model")        #saving the trained model

import pickle

#to save fitted tokenizer

with open('F:/pycache/chatbot/tokenizer.pickle', 'wb') as handle:   #Saving the tokenized files
    pickle.dump(tokenizer, handle,protocol =pickle.HIGHEST_PROTOCOL )
    
with open('F:/pycache/chatbot/label_encoder.pickle','wb') as ecn_file:
    pickle.dump(lbl_encoder , ecn_file,protocol=pickle.HIGHEST_PROTOCOL)

###############creating inference
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama
colorama.init( )
from colorama import Fore,Style,Back

import random
import pickle

with open("F:/pycache/chatbot/intents.json")as file:
    data = json.load(file)
    
    
#defining functions
def chat():
    model = keras.models.load_model('chat_model')
    
    with open ('F:/pycache/chatbot/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
    with open ('F:/pycache/chatbot/label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)
            max_len  =20
            
            
    while True :
                print(Fore.LIGHTBLUE_EX + "User :  "+Style.RESET_ALL,end = " ")
                inp = input( )
                if inp.lower( ) == "quit":
                    break
                result = model.predict(keras.preprocessing.sequence.pad_sequences
                                       (tokenizer.texts_to_sequences([inp]), truncating = 'post' , maxlen = max_len))
                tag = lbl_encoder.inverse_transform([np.argmax(result)])    #converts the given text into label
                
                for i in data['intents']:
                    if i['tag'] == tag:
                        print(Fore.GREEN + "Chat_Bot: " + Style.RESET_ALL, np.random.choice(i['responses'] ) )
print(Fore.YELLOW + " Start messaging the Bot is ready!!!!(press quit to stop)" + Style.RESET_ALL) 


chat( )    



