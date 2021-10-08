import pandas as pd
import numpy as np
import re
import time

import konlpy
from konlpy.tag import Okt

from pprint import pprint
import nltk

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model

import json
import os




class prepro:

    def __init__(self, data, model=0, token=0):
        self.model = model
        self.token = token
        self.data = data
    


    # 입력받은 testSet 형식 맞춰주기
    def dataPro(self):
        rv_test = pd.read_csv('Dataset/%s.csv' %self.data)
        rv_test = pd.DataFrame({'txt':rv_test['txt'], 'label':None})

        return rv_test

    

    # 정규 표현식을 사용해 한글만 남기기
    # 한글 자연어 분석을 위해 특수문자, 숫자, 영어를 삭제한다.
    # ㅋㅋㅋ, ㅎㅎㅎ 와 같은 문자는 감성분석이 아니기에 삭제한다.
    # 정규화를 통해 리스트에 저장된 문자열을 다시 문장 형태로 바꾸어준다.(konlpy를 통한 정규화를 위해)
    def regur(self, dataset):
        for i in range(len(dataset)):
            if type(dataset['txt'][i])==str: 
                convert = re.compile('[가-힣]+').findall(dataset['txt'][i])
                dataset['txt'][i] = ' '.join(convert)



    # konlpy 라이브러리에 적용하여 자연어 분석을 하기 위해
    # Dataframe에서 list형식으로 바꾸어 준다.
    def to_list(self, dataset):
        return dataset.values.tolist()



    # 결측값 처리
    # tokenize 처리시 결측 데이터를 ''뮨자열로 바꾸어 주어 처리한다.
    def del_none(self, dataset):
        for i in range(len(dataset)):
            if type(dataset[i][0])!=str:
                dataset[i][0]=''



    # 한글만 남은 리뷰 데이터에서
    # 불용어 제거, 어간 추출을 위해 konlpy의 okt라이브러리를 사용한다
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    def tokenize(self, doc):
        return ['/'.join(t) for t in Okt().pos(doc, norm=True, stem=True)]



    # 각 리뷰에 대한 토큰화 진행
    # 토큰화 한 후, 라벨과 함께 Json 파일로 저장
    def json_tokenize(self, list):
        secs = time.time()+32400
        tm = time.localtime(secs)
        time_log = '{0}-{1}-{2}-{3}-{4}-{5}_token'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
            
        # token_list : [토큰/stem] 형식의 list를 모아둔 1차원 list
        token_list = [(self.tokenize(row[0]), row[1]) for row in list]

        with open('Token/%s.json' %time_log, 'w', encoding='utf-8') as make_file:
            json.dump(token_list, make_file, ensure_ascii=False, indent='\t')

        return token_list

    

     # tokens : stem을 제외한 전체 토큰을 1차원 list로 나열한것 
     # 가장 많이 중복된 tokens 가져오기(1000개)
    def get_tokens_most(self, train_list):
        tokens = [t for d in train_list for t in d[0]]
        text = nltk.Text(tokens, name='리뷰데이터')
        selected_words = [f[0] for f in text.vocab().most_common(10000)]
        return selected_words


    # 중복된 횟수 count
    def term_frequency(self, doc, selected_words):
        return [doc.count(word) for word in selected_words]



    # 신경망 학습에 사용할 sparse data로 변환
    def sparse_data(self, list_data, selected_words):
        data_x = [self.term_frequency(d, selected_words) for d, _ in list_data]
        data_y = [c for _, c in list_data]
        return data_x, data_y



    # numpy를 사용해 array형식으로 바꾸어준다.
    def sparse_to_array(self, data_x, data_y):
        x_data = np.asarray(data_x).astype('float32')
        y_data = np.asarray(data_y).astype('float32')
        return x_data, y_data

    

    # train data를 이용한 모델 학습 및 저장
    def data_modeling(self, x_train, y_train):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                    loss=losses.binary_crossentropy,
                    metrics=[metrics.binary_accuracy])

        model.fit(x_train, y_train, epochs=10, batch_size=512)
        self.model = model
        
        secs = time.time()+32400
        tm = time.localtime(secs)
        time_log = '{0}-{1}-{2}-{3}-{4}-{5}_model'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)

        model.save('Model/%s.h5' %time_log)
    

    # test_set에 label 붙여 분류하기(1: 영화 리뷰, 0 : 그 외 리뷰)
    def give_label(self, review, models, selected_words):
        for i in range(len(review['txt'])):
            token = self.tokenize(review['txt'][i])
            tf = self.term_frequency(token, selected_words)
            data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
            score = float(models.predict(data))
            if(score > 0.5):
                review['label'][1] = 1
            else:
                review['label'][1] = 0        

        secs = time.time()+32400
        tm = time.localtime(secs)
        time_log = '{0}-{1}-{2}-{3}-{4}-{5}_Result'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)

        review.to_csv('Result/%s.csv' %time_log, sep=',', na_rep='NaN', encoding='utf-8-sig')
        


    # JSon 파일 리스트로 변환
    def json_toList(self):
        token_json = self.token

        with open('Token/%s.json' %token_json, 'r', encoding='utf-8') as st_json:
            st_python = json.load(st_json)
        return st_python


    # Model load 하기
    def loadModel(self, m_name):
        
        model = load_model('Model/%s.h5' %m_name)
        return model
