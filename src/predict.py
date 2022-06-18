# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 22:41
# @Author  : zxf
import os
import json

import torch

from common.common import logger
from utils.util import get_tokenizer
from models.model import SoftMaskedBertModel
from utils.util import data_inference_feature
from utils.util import processing_predict_result


def predict(text, pretrain_model_path, pretrain_model_type, model_path,
            max_length, hidden_size):
    device = "cpu"
    # tokenizer
    tokenizer = get_tokenizer(pretrain_model_path, pretrain_model_type)
    # model
    model = SoftMaskedBertModel(pretrain_model_path, hidden_size, pretrain_model_type,
                                tokenizer.mask_token_id, device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)),
                          strict=True)
    model.to(device)
    # get text feature
    input_token, mask, text_length, text_token = data_inference_feature(text, tokenizer, max_length)
    candidates = []
    with torch.no_grad():
        input_token = input_token.to(device)
        mask = mask.to(device)
        _, logits = model(input_token, input_token, mask)
        logits = logits[:, 1:text_length + 1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(logits)
        output_topk_prob = torch.topk(output_tensor, 1).values.squeeze(0).tolist()
        output_topk_indice = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()
        for i, index in enumerate(output_topk_indice):
            tmp = []
            for j, candidate in enumerate(index):
                word = tokenizer.convert_ids_to_tokens(candidate)
                tmp.append(word)
            candidates.append(tmp)

        result = processing_predict_result(candidates, text, text_token, max_length,
                                           output_topk_prob)
        print(result)


if __name__ == "__main__":
    text = "她告诉爱文：如果他在晚上喝酒他的生体一定不太好。"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/chinese-bert-wwm-ext/"
    pretrain_model_type = "bert"
    model_path = "./output/model.pt"
    max_length = 68
    hidden_size = 256
    predict(text, pretrain_model_path, pretrain_model_type, model_path,
            max_length, hidden_size)
