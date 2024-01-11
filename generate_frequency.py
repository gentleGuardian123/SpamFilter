import os
import time
import string
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from root_path import root_path

def generate_frequency():

    start_time = time.time()

    df = pd.read_csv(root_path + "wordlist.csv", header=0)
    # df = pd.read_csv(root_path + "test_wordlist.csv", header=0)
    words = df['word']

    lmtzr = WordNetLemmatizer()

    directory = os.fsencode(root_path + "emails/")

    f = open(root_path + "frequency.csv", "w+")
    # f = open(root_path + "test_frequency.csv", "w+")
    for i in words:
        f.write(str(i) + ',')
    f.write('output')
    f.write('\n')
    f.close()

    lines = []

    rd = 0
    # for file in os.listdir(directory)[:100]:
    for file in os.listdir(directory):
        file = file.decode("utf-8")
        file_name = root_path + 'emails/'
        for i in file:
            if(i != 'b' and i != "'"):
                file_name = file_name + i
        rd += 1
        file_reading = open(file_name,"r",encoding='utf-8', errors='ignore')
        words_list_array = np.zeros(words.size)
        for word in file_reading.read().split():
            word = lmtzr.lemmatize(word.lower())
            if word in stopwords.words('english') or word in string.punctuation or len(word) <= 2 or word.isdigit() == True:
                continue
            for i in range(words.size):
                if(words[i] == word):
                    words_list_array[i] = words_list_array[i] + 1
                    break
        line = ""
        for i in range(words.size):
            line += str(int(words_list_array[i])) + ','
        if(file.startswith('s')):
            line += "-1"
        elif (file.startswith('h')):
            line += "1"
        line += "\n"
        lines.append(line)
        if rd % 100 == 0:
            print("Finished email files: " + str(rd))
    
    random.shuffle(lines)

    f = open(root_path + "frequency.csv", "a")
    # f = open(root_path + "test_frequency.csv", "a")
    for line in lines:
        f.write(line)
    f.close()

    end_time = time.time()
    print("Time for segregating data and forming input vector(word frequency): " + str(round(end_time - start_time, 2)) + " seconds")
