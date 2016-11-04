""" a neat code from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/ """
import os

from utils.mc_data_utils import DataSet
from copy import deepcopy


import json
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def my_sent_tokenize(txt):
    tmp = sent_tokenize(txt)
    ret = []
    for t1 in tmp:
      if t1[0].islower():
        ret[len(ret)-1] = ret[len(ret)-1] + ' ' + t1
      else:
        ret.append(t1)
    return ret

def sent_to_tokens(sent, word_table):
    ret =  [stemmer.stem(lemmatizer.lemmatize(tmp)) for tmp in word_tokenize(sent.lower())]
    #print('token', ret)
    word_table.add_vocab(ret)
    return ret

def para_to_tokens(para, word_table):
    return [sent_to_tokens(sent, word_table) for sent in my_sent_tokenize(para)]




def read_mc(tt_type,batch_size,  word_table):
    if tt_type == 'train':
        data_dir = './data/traintest/train'
    elif tt_type == 'test':
        data_dir = './data/traintest/test'
    ret_data = []
    for fn in os.listdir(data_dir):
        fn_path = os.path.join(data_dir, fn)
        print('Load', tt_type , 'data from ', fn_path)
        td = json_get_data(fn_path,  word_table)
        ret_data.extend(td)
    s, q, a, l = zip(*ret_data)
    fc = [len(tmp) for tmp in s]
    return DataSet(batch_size, s, q, a, l, fc, name=tt_type)





def get_max_sizes(*data_sets):
    max_sent_size = max_ques_size = max_fact_count = max_answer_size = max_answer_count = 0
    for data in data_sets:
        for x, q, y, l, fc in zip(data.xs, data.qs, data.ys, data.ls, data.fact_counts):
            for fact in x: max_sent_size = max(max_sent_size, len(fact))
            max_ques_size = max(max_ques_size, len(q))
            max_fact_count = max(max_fact_count, fc)
            for aso in y: max_answer_size = max(max_answer_size, len(aso))
            max_answer_count = max(max_answer_count, len(y))

    return max_sent_size, max_ques_size, max_fact_count, max_answer_size, max_answer_count

def json_get_data(fname, word_table, label_num = 3):
    lbs = 'ABCDE'
    nums = '01234'
    dadict = dict(zip(lbs[:label_num], ['']*label_num))
    dldict = dict(zip(lbs[:label_num], nums[:label_num]))
    with open(fname) as f:
      lines = f.read()
    jlines = json.loads(lines)
    ret = []
    for exjson in jlines['exercises']:
      if 'text' not in exjson['story']:
        print("No 'text' in story")
        print(exjson)
        continue
      s_tokens = para_to_tokens(exjson['story']['text'], word_table)
      for qjson in exjson['questions']:
        q_tokens = sent_to_tokens(qjson['text'], word_table)
        a_dict = dadict.copy()
        for ansjson in qjson['answerChoices']:
          a_dict[ansjson['label']] = ansjson['text']
        a_tokens = []
        for lab in list(lbs[:label_num]):
          a_tokens.append(sent_to_tokens(a_dict[lab],word_table))
        l_dict = dldict.copy()
        if qjson['correctAnswer'] not in list(lbs[:label_num]):
            print("CorrectAnswer not in 'ABC'")
            print(qjson)
            continue
        true_ans = l_dict[qjson['correctAnswer']]
        ret.append([s_tokens, q_tokens, a_tokens, true_ans])
    return ret

