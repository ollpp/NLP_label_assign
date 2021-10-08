

#%%
import movieNlp 
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

okt = Okt()


data = input('예측하고자 하는 데이터파일명을 입력하세요 : ')
model = input('사용할 모델파일명을 입력하세요 : ')
token = input('사용할 토큰파일명을 입력하세요 : ')



nlp = movieNlp.prepro(data, model, token)


# 데이터 틀에 맞게 변형
rv_test = nlp.dataPro()

# test-data 정규화
nlp.regur(rv_test)

# to_list
test_list = nlp.to_list(rv_test)

# 결측값 처리
nlp.del_none(test_list)

# 불용어 제거, 어간 추출
test_docu = nlp.json_tokenize(test_list)

# 이미 train된 token 호출
train_docu = nlp.json_toList()

# 사전에 학습한 model 호출
models = nlp.loadModel(model)

# tokenize 및 중복된 단어 조회
selected_words = nlp.get_tokens_most(train_docu)



# # sparse data로 변형
train_x, train_y = nlp.sparse_data(train_docu, selected_words)
test_x, test_y = nlp.sparse_data(test_docu, selected_words)

# # 모델에 적용하기 위해 array형태로 변형 시켜주기
x_train, y_train = nlp.sparse_to_array(train_x, train_y)
x_test, y_test = nlp.sparse_to_array(test_x, test_y)

# 모델 학습 및 저장
nlp.data_modeling(x_train, y_train)

# test data에 labeling
nlp.give_label(rv_test, models, selected_words)

# %%
