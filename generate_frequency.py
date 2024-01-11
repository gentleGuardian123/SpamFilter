import root_path
import os
import string
import numpy as np
import pandas as pd
from time import time
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def generate_frequency():

    start_time = time()

    df = pd.read_csv(root_path + "wordslist.csv", header=0)
    words = df['word']

    lmtzr = WordNetLemmatizer()

    directory_in_str = "emails/"
    directory = os.fsencode(directory_in_str)

    f = open(root_path + "frequency.csv", "w+")
    for i in words:
        f.write(str(i) + ',')
    f.write('output')
    f.write('\n')
    f.close()

    k = 0
    for file in os.listdir(directory):
        file = file.decode("utf-8")
        file_name = str(os.getcwd()) + '/emails/'
        for i in file:
            if(i != 'b' and i != "'"):
                file_name = file_name + i
        k += 1
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
        f = open(root_path + "frequency.csv","a")
        for i in range(words.size):
            f.write(str(int(words_list_array[i])) + ',')
        if(file_name.startswith('s')):
            f.write("-1")
        elif (len(file_name) == 71):
            f.write("1")
        f.write('\n')
        f.close()
        if k % 100 == 0:
            print("Finished email files: " + str(k) * 100)

    print("Time for segregating data and forming input vector(word frequency): " + str(round(time() - start_time,2)) + " seconds")
