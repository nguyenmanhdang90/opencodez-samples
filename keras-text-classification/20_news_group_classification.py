import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
import argparse
import os.path

# For reproducibility
np.random.seed(1237)

def trainAndTest(data_path, num_label, vocab_size, batch_size, epoch, train_ratio):
    files_train = skds.load_files(data_path,load_content=False)
    label_index = files_train.target
    label_names = files_train.target_names
    labelled_files = files_train.filenames

    data_tags = ["filename","category","news"]
    data = readData(labelled_files, data_tags, label_names, label_index)

    # lets take 80% data as training and remaining 20% for test.
    train_size = int(len(data) * train_ratio)

    train_posts = data['news'][:train_size]
    train_tags = data['category'][:train_size]
    train_files_names = data['filename'][:train_size]

    test_posts = data['news'][train_size:]
    test_tags = data['category'][train_size:]
    test_files_names = data['filename'][train_size:]

    # define Tokenizer with Vocab Size
    tokenizer = loadVocab(data_path, vocab_size, num_label)
    tokenizer.fit_on_texts(train_posts)

    x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_test = encoder.transform(test_tags)

    model = loadModel(data_path, vocab_size, num_label)
    model.summary()

    train(train_posts, train_tags, tokenizer, encoder, batch_size, epoch, model)

    saveModel(model, data_path, vocab_size, num_label)

    saveVocab(tokenizer, data_path, vocab_size, num_label)

    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)

    print('Test accuracy:', score[1])

    text_labels = encoder.classes_

    for i in range(10):
        prediction = model.predict(np.array([x_test[i]]))
        predicted_label = text_labels[np.argmax(prediction[0])]
        print(test_files_names.iloc[i])
        print('Actual label:' + test_tags.iloc[i])
        print("Predicted label: " + predicted_label)
    pass

def readData(labelled_files, data_tags, label_names, label_index):
    data_list =[]
    for i,f in enumerate(labelled_files):
        data_list.append((f,label_names[label_index[i]],Path(f).read_text(encoding='cp1252')))
    return pd.DataFrame.from_records(data_list, columns=data_tags)


def train(train_posts, train_tags, tokenizer, encoder, batch_size, epoch, model):
    
    x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')

    y_train = encoder.transform(train_tags)

    model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epoch,
                verbose=1,
                validation_split=0.1)

def saveModel(model, data_path, vocab_size, num_label):
    fileName = data_path + '/' + '_'.join([str(vocab_size), str(num_label)])
    model.save(fileName)

def loadModel(data_path, vocab_size, num_label):
    fileName = data_path + '/' + '_'.join([str(vocab_size), str(num_label)])
    if(os.path.isfile(fileName)):
        return load_model(fileName)
    return createModel(vocab_size, num_label)

def createModel(vocab_size, num_label):
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_label))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model
    

def saveVocab(tokenizer, data_path, vocab_size, num_label):
    fileName = data_path + '/' + '_'.join([str(vocab_size), str(num_label), 'tokenizer'])
    with open(fileName, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadVocab(data_path, vocab_size, num_label):
    fileName = data_path + '/' + '_'.join([str(vocab_size), str(num_label), 'tokenizer'])
    if(os.path.isfile(fileName)):
        with open(fileName, 'rb') as handle:
            return pickle.load(handle)
    return Tokenizer(num_words=vocab_size)

class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description = '')
        parser.add_argument("-dp", "--data_path", help = "data path", required = False, default = './data/20news-bydate/20news-bydate-train')
        parser.add_argument("-nl", "--num_label", help = "number of label", required = False, default = 20)
        parser.add_argument("-vs", "--vocab_size", help = "number of vocabulary", required = False, default = 15000)
        parser.add_argument("-bs", "--batch_size", help = "batch size", required = False, default = 100)
        parser.add_argument("-tr", "--train_ratio", help = "train vs test ratio", required = False, default = .8)
        parser.add_argument("-ep", "--epoch", help = "epoch number", required = False, default = 20)
        
        argument = parser.parse_args()
        print(argument.data_path)
        print(argument.num_label)
        print(argument.vocab_size)
        print(argument.train_ratio)
        trainAndTest(
            argument.data_path,
            int(argument.num_label),
            int(argument.vocab_size),
            int(argument.batch_size),
            int(argument.epoch), 
            float(argument.train_ratio))

def main(argv):
    print(argv)
    

if __name__== "__main__":
    # path_train = "./data/20news-bydate/20news-bydate-train"
    # train(path_train)
    app = CommandLine()