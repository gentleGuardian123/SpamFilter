# import json
import os
import time
import string
import operator
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from root_path import root_path

def text_cleanup(text):
    text_without_punctuation = [c for c in text if c not in string.punctuation]
    text_without_punctuation = ''.join(text_without_punctuation)
    text_without_stopwords = [word for word in text_without_punctuation.split() if word.lower() not in stopwords.words('english')]
    text_without_stopwords = ' '.join(text_without_stopwords)
    cleaned_text = [word.lower() for word in text_without_stopwords.split()]
    return cleaned_text

def generate_wordlist():
    start_time = time.time()
    lmtzr = WordNetLemmatizer()
    rd = 0
    directory = os.fsencode(root_path + "emails/")
    dc_dict = {}  # 统计词语出现的文档数
    tf_sum_dict = {}  # 统计词语的在各文档的tf总和
    # for file in os.listdir(directory)[:100]:
    for file in os.listdir(directory):
        file = file.decode("utf-8")
        file_name = root_path + "emails/" + file
        file_reading = open(file_name,"r",encoding='utf-8', errors='ignore')
        count = {}
        words = text_cleanup(file_reading.read())
        word_len = float(len(words))
        for word in words:
            if (word.isdigit() == False and len(word) > 2):
                word = lmtzr.lemmatize(word)
                try:
                    count[word] += 1
                except:
                    count[word] = 1
                    try:
                        dc_dict[word] += 1
                    except:
                        dc_dict[word] = 1
        # json.dump(count, open("路径", 'w'))  这里可以用json.dump保存各个邮件的词汇字典（含数量），可以避免后学根据特征词构造特征向量重新进行分词查询
        # json.load(open("路径", 'r'))  通过load可以从文件种读取python对象，后续查询可直接根据字典中是否存在对应值查询
        for (key, value) in count.items():
            try:
                tf_sum_dict[key] += float(value) / word_len
            except:
                tf_sum_dict[key] = float(value) / word_len
        rd += 1
        if(rd % 100 == 0):
            print("Finished email file: " + str(rd))

    # 开始计算各词汇的tf-idf值，并构造字典
    tf_idf_dict = {}
    for (key, tf_value) in tf_sum_dict.items():
        tf_idf_dict[key] = tf_value * (1 / float(dc_dict[key]))   # 也可以做log处理
    # 排序挑选word_number数量的特征词
    sorted_count = sorted(tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True)
    word_number = 900                               # 特征词的数量，可自行调整
    f = open(root_path + "wordlist.csv", "w+")
    # f = open(root_path + "test_wordlist.csv", "w+")
    f.write('word,count')
    f.write('\n')
    for (word, tf_idf_value) in sorted_count[:word_number]:
        f.write(str(word) + ',' + str(tf_idf_value))
        f.write('\n')
    f.close()
    end_time = time.time()
    print('Time spent on preprocessing emails and generting wordlist: ' + str(round(end_time - start_time, 2)) + ' seconds')
