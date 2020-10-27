import sys
import os
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from tqdm import tqdm

# 참고한 튜토리얼 : https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb?fbclid=IwAR1CiTt_tKSvh4ee_Kpep41yS8Dhd6m9osJYZaRaR5qFuycOvADeCK6jIZA#scrollTo=zVvslsfMIrIh

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix

# 1. dataset 준비
# local file loading 매뉴얼 참조 : https://huggingface.co/docs/datasets/loading_datasets.html?fbclid=IwAR2om0_gpIbQ07vK0Rhy1fhpaOSk5cWJAC-vxcx269Di2SXDPefR7x9hZ3M#from-local-files
#   - text data를 읽어서 dataset을 만든다.
#   - train, valid, test는 기본모델 실험과 동일하게 순서대로 8:1:1로 분할해서 만든다. 
#   - label은 숫자로 변환하고, 참조할 사전을 만들어둔다.

train_data = {'idx': [], 'label': [], 'sentence': []}
valid_data = {'idx': [], 'label': [], 'sentence': []}
test_data  = {'idx': [], 'label': [], 'sentence': []}
input_path = './SMSSpamCollection.txt'
label2id = {}
g_labelid = 0
id2label = {}
tot_num_line = sum(1 for _ in open(input_path, 'r'))
train_num = tot_num_line * 0.8
valid_num = train_num + tot_num_line * 0.1
with open(input_path, 'r', encoding='utf-8') as f:
    train_idx = 0
    valid_idx = 0
    test_idx  = 0
    for idx, line in enumerate(tqdm(f, total=tot_num_line)):
        line = line.strip()
        tokens = line.split('\t')
        label = tokens[0]
        if label not in label2id:
            label2id[label] = g_labelid
            id2label[g_labelid] = label
            g_labelid += 1
        labelid = label2id[label]
        sentence = tokens[1]
        if idx <= train_num:                       # 8/10 train
            data = train_data
            seq = train_idx
            train_idx += 1
        elif idx > train_num and idx <= valid_num: # 1/10 valid
            data = valid_data
            seq = valid_idx
            valid_idx += 1
        else:                                      # 1/10 test
            data = test_data
            seq = test_idx
            test_idx += 1
        data['idx'].append(seq)
        data['label'].append(labelid)
        data['sentence'].append(sentence)
train_dataset = Dataset.from_dict(train_data)
valid_dataset = Dataset.from_dict(valid_data)
test_dataset = Dataset.from_dict(test_data)

# 2. metric 설정
#   - transformers Trainer를 사용하려면 'metric'이 필요
metric = load_metric('glue', 'sst2')

# 3. pretrained model 설정
#   - 데이터가 대소문자 구별하고 있어서 'cased' 사용

model_checkpoint = 'distilbert-base-cased'
#model_checkpoint = 'bert-base-cased'

# 4. 데이터를 tokenizer를 이용해서 전처리
#   - transformers model을 위해서 dataset을 전처리해줘야한다.

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding=True, truncation=True)
encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)

# 5. 모델 생성
#   - classification 문제이므로 이미 있는 모델을 가져다 사용할 수 있다.

num_labels = len(label2id)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# 6. TrainingArguments 생성

task = 'spamfilter'
metric_name = 'eval_loss'
args = TrainingArguments(
    task,
    evaluation_strategy = 'epoch',
    learning_rate=1e-5,
    per_device_train_batch_size=16, # distil-bert-*: 16, bert-*: 32
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

# 7. metric 연결하고 Trainer 생성

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
   
    # classification report 출력
    target_names = [v for k, v in sorted(id2label.items(), key=lambda x: x[0])] 
    print(classification_report(labels, predictions, target_names=target_names, digits=4)) 
    print(confusion_matrix(labels, predictions))
    
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 8. 학습 및 평가 실시

trainer.train()
print('')
trainer.predict(encoded_test_dataset)
