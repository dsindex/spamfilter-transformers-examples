"""
reference : https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb?fbclid=IwAR1CiTt_tKSvh4ee_Kpep41yS8Dhd6m9osJYZaRaR5qFuycOvADeCK6jIZA#scrollTo=zVvslsfMIrIh
"""

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

from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./train.log')
logger.addHandler(streamHandler)
logger.addHandler(fileHandler)

def prepare_dataset(input_path):
    train_data = {'idx': [], 'label': [], 'sentence': []}
    valid_data = {'idx': [], 'label': [], 'sentence': []}
    test_data  = {'idx': [], 'label': [], 'sentence': []}
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

    return train_dataset, valid_dataset, test_dataset, label2id, id2label

def get_params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='spamfilter')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss')
    parser.add_argument('--input_path', type=str, default='./SMSSpamCollection.txt')
    parser.add_argument('--bert_model_name_or_path', type=str, default='distilbert-base-cased')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--seed', default=42, type=int)

    opt = parser.parse_args()
    return opt

def main():
    opt = get_params()

    # prepare / preprocess data
    train_dataset, valid_dataset, test_dataset, label2id, id2label = prepare_dataset(opt.input_path)
    logger.info("dataset ready")

    tokenizer = AutoTokenizer.from_pretrained(opt.bert_model_name_or_path, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding=True, truncation=True)
    encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
    encoded_valid_dataset = valid_dataset.map(preprocess_function, batched=True)
    encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)
    logger.info("preprocessing done")

    # prepare model
    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(opt.bert_model_name_or_path, num_labels=num_labels)
    logger.info("classification model ready")

    # prepare trainer
    args = TrainingArguments(
        opt.task,
        evaluation_strategy = 'epoch',
        learning_rate=opt.learning_rate,
        per_device_train_batch_size=opt.per_device_train_batch_size, # distil-bert-*: 16, bert-*: 32
        per_device_eval_batch_size=opt.per_device_eval_batch_size,
        num_train_epochs=opt.num_train_epochs,
        weight_decay=opt.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=opt.metric_for_best_model,
    )

    metric = load_metric('glue', 'sst2')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # classification report
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

    # train
    trainer.train()

    # predict
    trainer.predict(encoded_test_dataset)


if __name__ == '__main__':
    main()
