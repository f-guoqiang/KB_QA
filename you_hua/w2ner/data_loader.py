import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import pickle
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            # 使用自增1的方式进行
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

    def save(self):
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(self.label2id, f)
            pickle.dump(self.id2label, f)

    def load(self):
        with open('vocab.pkl', 'rb') as f:
            data = pickle.load(f)
        return data


# pytoch会调用这个函数处理数据
def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    # 得到最大的长度
    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)

    max_pie = np.max([x.shape[0] for x in bert_inputs])
    # 对batch进行填充
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)

    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


# vocab存的是标签到数字的映射
def process_bert(data, tokenizer, vocab):
    # 这些变量都是用来存什么的
    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        # 分成字词（还是英文的，不是数字），分成了子词
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        # 将这句话的子词放在一起,把token处理成[[],[]]-->[,,]
        pieces = [piece for pieces in tokens for piece in pieces]
        # 转换成数字
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        # 加cls和sep tokenizer.cls_token_id
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        length = len(instance['sentence'])
        if length > 200:
            continue

        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                # 标记这个词的所在的位置
                pieces = list(range(start, start + len(pieces)))
                # 将对应位置设置成1，标记piece在哪里，这是个2维矩阵
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]

        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            # 这个会重复？
            if len(index) == 0:
                continue
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                # 实体的位置标注成1
                _grid_labels[index[i], index[i + 1]] = 1
            # 将最后一个位置
            # list index out of range,原因是实体内的数字是0?
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


# dataset 是训练数据
def fill_vocab(vocab, dataset):
    entity_num = 0  # 不考虑重复
    for instance in dataset:
        for entity in instance["ner"]:
            # vocab类实现的函数方法add_label
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


from sklearn.model_selection import train_test_split


def load_data_bert(config):
    with open(r'F:\GitHub\W2NER\data\CME\train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(r'F:\GitHub\W2NER\data\CME\dev.json', 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    # 加载没有label的测试集
    # with open(r'F:\GitHub\W2NER\data\CME\test.json', 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)
    # 在这里加在的
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir="./cache/")
    # 实现label2id,id2label
    vocab = Vocabulary()
    # 返回的是实体的个数
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    # test_ent_num = fill_vocab(vocab, test_data)

    # 打印信息的方法
    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    # table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    vocab.save()

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    # test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    # return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
    dev_dataset, test_dataset = train_test_split(dev_dataset, test_size=0.33, random_state=42)
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data)


def load_text_data_bert(config):
    with open(r'F:\GitHub\W2NER\data\CME\test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext", cache_dir="./cache/")
        # 实现label2id,id2label
    vocab = Vocabulary()
    vocab.load()
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return test_dataset
