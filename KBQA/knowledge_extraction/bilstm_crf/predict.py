# coding=utf-8
import keras
import numpy as np
from bilstm_crf_model import BiLstmCrfModel
from crf_layer import CRF
from data_helpers import NerDataProcessor
import tensorflow as tf
import os

max_len = 80
vocab_size = 2410
embedding_dim = 200
lstm_units = 128

if __name__ == '__main__':

    ndp = NerDataProcessor(max_len, vocab_size)

    test_X, test_y = ndp.read_data(
        "../../../ChineseBLUE/data/cMedQANER/test.txt",
        is_training_data=True
    )
    test_X, test_y = ndp.encode(test_X, test_y)

    class_nums = ndp.class_nums
    word2id = ndp.word2id
    tag2id = ndp.tag2id
    id2tag = ndp.id2tag
    from keras.models import load_model
    from crf_layer import CRF
    from metrics import *

    path = r"F:\GitHub\KBQA\KBQA\knowledge_extraction\bilstm_crf\checkpoint\best_bilstm_crf_model.h5"

    model = load_model(path, custom_objects={"CRF": CRF,"loss":classification_report})

    pred = model.predict(test_X)

    # 求实体的准确性，而不是模型中输入的每个token的准确性
    from metrics import *

    y_true, y_pred = [], []

    for t_oh, p_oh in zip(test_y, pred):
        # 换标签中的症状
        t_oh = np.argmax(t_oh, axis=1)
        t_oh = [id2tag[i].replace('_', '-') for i in t_oh if i != 0]
        p_oh = np.argmax(p_oh, axis=1)
        p_oh = [id2tag[i].replace('_', '-') for i in p_oh if i != 0]

        y_true.append(t_oh)
        y_pred.append(p_oh)

    f1 = f1_score(y_true, y_pred, suffix=False)
    p = precision_score(y_true, y_pred, suffix=False)
    r = recall_score(y_true, y_pred, suffix=False)
    acc = accuracy_score(y_true, y_pred)

    print(
        "f1_score: {:.4f}, precision_score: {:.4f}, recall_score: {:.4f}, accuracy_score: {:.4f}".format(f1, p, r, acc))
    print(classification_report(y_true, y_pred, digits=4, suffix=False))