# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 20:27
# @Author  : zxf
import os
import ast
import json
import random
import operator
from copy import deepcopy

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertTokenizer


class SoftMaskedBertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        super(SoftMaskedBertDataset, self).__init__()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        df = self.data.iloc[idx]
        error_text = df['random_text'].strip()
        correct_text = df['origin_text'].strip()
        # # 对错字位置 one-hot 进行预处理，将字符串转为list
        error_label = df['label'].strip().split()
        error_label = [int(item) for item in error_label]
        # print("idx: ", idx)
        # print("error_text: ", error_text)
        # print("correct_text: ", correct_text)
        error_text = ' '.join(error_text.split()).replace(' ', '✈')
        correct_text = ' '.join(correct_text.split()).replace(' ', '✈')
        # print("idx: ", idx)
        # print("error_text: ", error_text)
        # print("correct_text: ", correct_text)
        error_text = list(error_text)
        correct_text = list(correct_text)

        assert len(error_text) == len(correct_text) == len(error_label), \
            f'error text size:{len(error_text)}, ' \
            f'correct_text size:{len(correct_text)},' \
            f'error label size:{len(error_label)}'

        error_text_ids = []
        correct_text_ids = []
        for i in range(len(error_text)):
            temp_error_text = self.tokenizer(error_text[i], add_special_tokens=False)['input_ids']
            temp_correct_text = self.tokenizer(correct_text[i], add_special_tokens=False)['input_ids']
            error_text_ids.extend(temp_error_text)
            correct_text_ids.extend(temp_correct_text)

        # error_text_ids = self.tokenizer(error_text, add_special_tokens=False)['input_ids']
        # correct_text_ids = self.tokenizer(correct_text, add_special_tokens=False)['input_ids']
        if len(error_text_ids) >= self.max_length - 2:
            error_text_ids = error_text_ids[:self.max_length - 2]
            correct_text_ids = correct_text_ids[:self.max_length - 2]
            error_label = error_label[:self.max_length - 2]

        error_text_ids = [self.cls_token_id] + error_text_ids + [self.sep_token_id]
        correct_text_ids = [self.cls_token_id] + correct_text_ids + [self.sep_token_id]
        error_label = [0] + error_label + [0]

        assert len(error_text_ids) == len(correct_text_ids) == len(error_label), \
            f'error_text_ids size:{len(error_text_ids)},' \
            f'correct_text_ids size:{len(correct_text_ids)},' \
            f'error_label size:{len(error_label)}'

        mask = [1] * len(error_label)
        padding_length = self.max_length - len(error_label)
        error_text_ids = error_text_ids + [self.pad_token_id] * padding_length
        correct_text_ids = correct_text_ids + [self.pad_token_id] * padding_length
        error_label = error_label + [0] * padding_length
        mask = mask + [0] * padding_length

        return {"error_text_ids": error_text_ids,
                "correct_text_ids": correct_text_ids,
                "error_label": error_label,
                "mask": mask}


def collate_fn(batch_data):
    batch_error_text = [item["error_text_ids"] for item in batch_data]
    batch_correct_text = [item["correct_text_ids"] for item in batch_data]
    error_label = [item["error_label"] for item in batch_data]
    mask = [item["mask"] for item in batch_data]

    batch_error_text = torch.tensor(batch_error_text, dtype=torch.long)
    batch_correct_text = torch.tensor(batch_correct_text, dtype=torch.long)
    error_label = torch.tensor(error_label, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    return {"error_text": batch_error_text,
            "correct_text": batch_correct_text,
            "error_label": error_label,
            "mask": mask}


def model_evaluate(dev_dataloader, model, device):
    model.eval()
    acc_num = 0
    total_num = 0
    word_acc_num = 0
    abs_acc_num = 0
    detector_num = 0
    for index, batch_data in enumerate(dev_dataloader):
        batch_input = batch_data["error_text"].to(device)
        batch_out = batch_data["correct_text"].to(device)
        batch_labels = batch_data["correct_text"].to(device)# .data.cpu().numpy().tolist()
        batch_mask = batch_data["mask"].to(device)
        batch_detect_labels = batch_data["error_label"].data.cpu().numpy().tolist()[0]
        length = torch.sum(batch_mask, dim=1).data.cpu().numpy().tolist()[0] - 2
        batch_labels = batch_labels.data.cpu().numpy().tolist()
        prob, output = model(batch_input, batch_out, batch_mask)
        # output = output[:, 1:-1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(output)
        output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
        pred_ = torch.argmax(output_tensor, dim=-1).tolist()

        # 统计句子级准确率， todo recall, precision, f1_score 如何计算
        pred_abs_result = operator.eq(pred_[0][:int(length) + 2][1:-1], batch_labels[0][:int(length) + 2][1:-1])
        if pred_abs_result:
            abs_acc_num += 1
        # 统计句子绝对准确率,只考虑detector预测的错字是否对，不考虑句子其他的字
        # 按照整个预测句子纠正后的句子和真实的句子比较,不能按照detector得到错误位置的句子词和真实的句子词直接比较
        prob = torch.round(prob)
        prob = torch.round(prob).squeeze() * batch_mask
        prob = prob.data.cpu().numpy().tolist()[0]
        # cls, sep 的位置为0
        prob[0] = 0
        prob[int(length) + 1] = 0
        detector_pred_index = [i for i in range(len(prob)) if prob[i] == 1]
        # pred_
        pred_batch_input = batch_input.clone().data.cpu().numpy().tolist()[0]
        if detector_pred_index:
            for item in detector_pred_index:
                pred_batch_input[item] = pred_[0][item]
        pred_result = operator.eq(pred_batch_input, batch_labels[0])
        if pred_result:
            acc_num += 1
        # detector acc
        dector_result = operator.eq(prob, batch_detect_labels)
        if dector_result:
            detector_num += 1
        # 统计词级别的acc
        # total_num += len(batch_labels[0][1:-1])
        # for j in range(len(batch_labels[0][1:-1])):
        #     if batch_labels[0][1:-1][j] == pred_[0][j]:
        #         word_acc_num += 1

        total_num += length
        for j in range(int(length)):
            if batch_labels[0][:int(length)+2][1:-1][j] == pred_[0][:int(length)+2][1:-1][j]:
                word_acc_num += 1

    acc = float(acc_num / len(dev_dataloader))
    word_acc = float(word_acc_num / total_num)
    abs_total_acc = float(abs_acc_num / len(dev_dataloader))
    detector_acc = float(detector_num) / len(dev_dataloader)
    return acc, abs_total_acc, word_acc, detector_acc


def model_evaluate_v2(dev_dataloader, model, device):
    model.eval()
    total_num = 0
    total_detector_acc = 0
    total_correct_acc = 0
    sent_acc_num = 0
    sent_detector_num = 0
    for index, batch_data in enumerate(dev_dataloader):
        batch_input = batch_data["error_text"].to(device)
        batch_out = batch_data["correct_text"].to(device)
        batch_labels = batch_data["correct_text"].to(device)# .data.cpu().numpy().tolist()
        batch_mask = batch_data["mask"].to(device)
        batch_error_label = batch_data["error_label"].to(device)
        length = torch.sum(batch_mask, dim=1).data.cpu().numpy().tolist()[0] - 2
        data_char_num = np.sum([len(item) for item in batch_mask])
        # batch_labels = batch_labels.data.cpu().numpy().tolist()
        prob, output = model(batch_input, batch_out, batch_mask)
        output_tensor = torch.nn.Softmax(dim=-1)(output)
        output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
        pred_ = torch.argmax(output_tensor, dim=-1)
        # 统计句子级准确率， todo recall, precision, f1_score 如何计算
        # pred_result = operator.eq(pred_[0][1:-1][:int(length)], batch_labels[0][1:-1][:int(length)])

        # 统计句子绝对准确率,只考虑detector预测的错字是否对，不考虑句子其他的字
        # 按照整个预测句子纠正后的句子和真实的句子比较,不能按照detector得到错误位置的句子词和真实的句子词直接比较

        # detector acc
        prob = torch.round(prob)
        # detector_acc = \
        #     np.sum([(prob.squeeze() * batch_mask).reshape(-1)[i].equal((batch_error_label * batch_mask).reshape(-1)[i])
        #             for i in range(len(prob.reshape(-1)))])
        # total_detector_acc += detector_acc
        prob = prob.squeeze() * batch_mask
        for i in range(len(prob)):
            temp = prob[i].equal(batch_error_label[i] * batch_mask[i])
            if temp:
                sent_detector_num += 1

        # correct acc, 按照字符级别计算
        correct_acc = np.sum([(batch_out * batch_mask).reshape(-1)[i].equal((pred_ * batch_mask).reshape(-1)[i])
                              for i in range(len(pred_.reshape(-1)))])
        total_correct_acc += correct_acc
        total_num += data_char_num
        # 按照句子级别计算acc
        for i in range(len(pred_)):
            temp_pred = pred_[i] * batch_mask[i]
            temp_label = batch_labels[i] * batch_mask[i]
            temp_result = temp_pred.equal(temp_label)
            if temp_result:
                sent_acc_num += 1

    word_correct_acc = np.float(total_correct_acc / total_num)
    # word_detector_acc = np.float(total_detector_acc / total_num)
    detector_acc = float(sent_detector_num) / len(dev_dataloader)
    abs_acc = float(sent_acc_num / len(dev_dataloader))
    return word_correct_acc, detector_acc, abs_acc


def get_tokenizer(pretrain_model_path, model_type):
    """
    :param pretrain_model_path: string
    :param model_type: string
    :return:
    获取tokenizer
    """
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    return tokenizer


def token_process(token_id, token_mask_id):
    """
    以80%的几率替换为[MASK]，以10%的几率保持不变，
    以10%的几率替换为一个随机token。
    """
    rand = np.random.random()
    if rand <= 0.75:
        return token_mask_id
    elif 0.9 >= rand >= 0.75:
        return token_id
    else:
        # 这是所有汉字的编号范畴，仅从汉字中抽取替换字符
        return np.random.randint(672, 7992)


def isChineseWord(string):
    result = []
    if string.isalpha():
        for i, item in enumerate(string):
            if ord(item) in range(65, 91) or ord(item) in range(97,123) :
                # return False
                result.append(False)
            else:
                # return True
                result.append(True)
        if False in result:
            return False
        else:
            return True
    return False


def word_ids_one_mask(texts_ids, text_tokens, mask_rate, mask_token_id):
    tmp_ids = []
    tmp_masks = []
    tmp_isG = []
    # 为每个字或者词生成一个概率，用于判断是否mask
    mask_rates = np.random.random(len(texts_ids))

    for i, word_id in enumerate(texts_ids):
        # 为每个字生成对应概率
        # tmp_ids.append(word_id)
        # 判断word_id 对应的token 是否为汉字
        if isChineseWord(text_tokens[i]):
            if mask_rates[i] < mask_rate:
                temp_texts_ids = deepcopy(texts_ids)
                temp_text_tokens = deepcopy(text_tokens)
                temp_texts_ids.pop(i)
                temp_text_tokens.pop(i)
                index = [item for item in range(len(temp_text_tokens))]
                # sample_word_id = random.sample(temp_texts_ids, 1)[0]
                sample_index = random.sample(index, 1)[0]
                sample_word_id = temp_texts_ids[sample_index]
                sample_temp_token = temp_text_tokens[sample_index]
                if isChineseWord(sample_temp_token):
                    tmp_masks.append(token_process(sample_word_id, mask_token_id))
                    tmp_isG.append(1)
                else:
                    tmp_masks.append(word_id)
                    tmp_isG.append(0)
            else:
                tmp_masks.append(word_id)
                tmp_isG.append(0)
        else:
            tmp_masks.append(word_id)
            tmp_isG.append(0)
    # tmp_ids = [cls_token_ids] + tmp_ids + [sep_token_ids]
    # tmp_masks = [0] + tmp_masks + [0]
    # instances.append([tmp_ids, tmp_masks, tmp_isG])
    return tmp_masks, tmp_isG


def get_random_wrong_data(data, tokenizer, mask_rate):
    """随机mask数据，造错误的数据"""
    random_mask_data = []
    for sent in data:
        # tokens = tokenizer.convert_tokens_to_ids(list(sent))
        sent = sent.lower().replace('#', '')
        sent = ''.join(sent.split())
        tokens_ids = tokenizer(sent, add_special_tokens=False)['input_ids']
        tokens = tokenizer.tokenize(sent)
        # mask shuju
        mask_tokens, mask_posi = word_ids_one_mask(tokens_ids, tokens,
                                                   mask_rate, tokenizer.mask_token_id)
        mask_sent = tokenizer.convert_ids_to_tokens(mask_tokens)
        random_mask_data.append([sent, ''.join(mask_sent).replace("#", ''),
                                 ' '.join([str(item) for item in mask_posi])])
    return random_mask_data


def data_inference_feature(text, tokenizer, max_length):
    """
    :param text: string
    :param tokenizer: tokenizer
    :param max_length: int
    :return:
    对预测文本进行模型推理
    """
    # tokens = tokenizer.convert_tokens_to_ids(list(text))
    text_tokens = tokenizer.tokenize(text)
    tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    if len(tokens) >= (max_length - 2):
        tokens = tokens[:max_length - 2]
        text_tokens = tokens[:max_length - 2]
    tokens = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
    length = len(tokens)
    mask = [1] * len(tokens)
    padding_length = max_length - len(tokens)
    tokens = tokens + [tokenizer.pad_token_id] * padding_length
    mask = mask + [0] * padding_length
    input_tokens = torch.tensor([tokens], dtype=torch.long)
    mask = torch.tensor([mask], dtype=torch.float32)
    return input_tokens, mask, length, text_tokens


def processing_predict_result(candidates, text, tokens, max_length, probs):
    # text_list = list(text)[:max_length - 2]
    correct_sentence = []
    result = {
        '原句': text,
        '纠正': '',
        '纠正数据': [
        ]
    }

    for i, ori in enumerate(tokens):
        if ori == candidates[i][0]:
            correct_sentence.append(ori)
            continue
        correct = {}
        correct['原字'] = ori
        candidate = candidates[i]
        confidence = probs[i]
        tmp_can = []
        tmp_cof = []
        for index, score in enumerate(confidence):
            if score > 0.001:
                tmp_can.append(candidate[index])
                tmp_cof.append(confidence[index])
        if ori in tmp_can:
            correct_sentence.append(ori)
            continue
        if confidence[0] > 0.99:
            correct['新字'] = candidate[0]
            correct['候选字'] = candidate
            result['纠正数据'].append(correct)
            correct_sentence.append(candidate[0])
        else:
            correct_sentence.append(ori)

    result['纠正'] = ''.join(correct_sentence).replace('#', '')
    return result


def model_evaluate_acc_prf(model, dev_dataloader, device):
    model.eval()
    detector_num = 0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    for index, batch_data in enumerate(dev_dataloader):
        batch_input = batch_data["error_text"].to(device)
        batch_out = batch_data["correct_text"].to(device)
        batch_correct = batch_data["correct_text"].to(device)# .data.cpu().numpy().tolist()
        batch_mask = batch_data["mask"].to(device)
        batch_detect_labels = batch_data["error_label"].data.cpu().numpy().tolist()[0]
        length = torch.sum(batch_mask, dim=1).data.cpu().numpy().tolist()[0]
        batch_correct = batch_correct.data.cpu().numpy().tolist()
        prob, output = model(batch_input, batch_out, batch_mask)
        # output = output[:, 1:-1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(output)
        output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
        pred_ = torch.argmax(output_tensor, dim=-1).tolist()
        # detector 预测的位置
        prob = torch.round(prob)
        prob = torch.round(prob).squeeze() * batch_mask
        prob = prob.data.cpu().numpy().tolist()[0]
        # cls, sep 的位置为0
        prob[0] = 0
        prob[int(length) - 1] = 0
        dector_result = operator.eq(prob, batch_detect_labels)
        if dector_result:
            detector_num += 1
        # 计算 acc 和prf
        correct = batch_correct[0][:int(length)][1:-1]
        pred_result = pred_[0][:int(length)][1:-1]
        error = batch_data["error_text"].data.cpu().numpy().tolist()[0][:int(length)][1:-1]
        # 没有错字的样本为反例
        if operator.eq(correct, error):
            # 预测也为负
            if operator.eq(correct, pred_result):
                TN += 1
            # 预测为正
            else:
                FP += 1
        else:
            # 预测也为正
            if operator.eq(correct, pred_result):
                TP += 1
            # 预测为负
            else:
                FN += 1

    acc = (TP + TN) / len(dev_dataloader)
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    detector_acc = float(detector_num / len(dev_dataloader))
    return detector_acc, acc, precision, recall, f1