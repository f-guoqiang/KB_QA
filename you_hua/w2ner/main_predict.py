import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader_predict
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        # 为不同的参数分配不同的学习率
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # updates_total = 数据集 / batch_size * epoch
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            # 最后一个是entity_text
            data_batch = [data.cuda() for data in data_batch[:-1]]
            #弄清作者这几个变量是用来干什么的
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            grid_mask2d = grid_mask2d.clone()
            # 计算CrossEntropyLoss
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list.append(loss.cpu().item())
            outputs = torch.argmax(outputs, -1)

            # contiguous强制拷贝一份grid_labels[grid_mask2d]
            # 修改后的量的改变不会影响之前的
            # view是改变矩阵形状，-1表示的是改变形状成一维度的
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels)
            pred_result.append(outputs)

            self.scheduler.step()
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def predict(self, epoch, data_loader):
        self.model.eval()
        pred_result = []
        result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                # sentence_batch = data[i:i+config.batch_size]
                data_batch = [data.cuda() for data in data_batch]
                bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                return outputs
        #         ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())
        #
        #         for ent_list, sentence in zip(decode_entities, sentence_batch):
        #             sentence = sentence["sentence"]
        #             instance = {"sentence": sentence, "entity": []}
        #             for ent in ent_list:
        #                 instance["entity"].append({"text": [sentence[x] for x in ent[0]],
        #                                            "type": config.vocab.id_to_label(ent[1])})
        #             result.append(instance)
        #
        #         total_ent_r += ent_r
        #         total_ent_p += ent_p
        #         total_ent_c += ent_c
        #
        #         grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
        #         outputs = outputs[grid_mask2d].contiguous().view(-1)
        #
        #         pred_result.append(outputs)
        #         i += config.batch_size
        #
        # pred_result = torch.cat(pred_result)
        #
        # with open(config.predict_path, "w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/example_bio.json')
    parser.add_argument('--save_path', type=str, default='./mc-bert/bio_model_2.pt')
    parser.add_argument('--predict_path', type=str, default='./output_bio_2.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger
    config.label_num = 11

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    # dataset返回的是元组(train_dataset, dev_dataset, test_dataset)
    # 对于CMEEE，test内的数据是空的，所以应该

    datasets = data_loader_predict.load_text_data_bert(config)
    test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader_predict.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )
    # 先除再乘
    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info("Building Model")
    model = Model(config)
    model = model.cuda()
    model.load_state_dict(r'F:\GitHub\W2NER\mc-bert\bio_model_2.pt')
    trainer = Trainer(model)
    output = trainer.predict(0, test_loader)
    print(output)