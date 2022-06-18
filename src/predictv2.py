# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 23:04
# @Author  : zxf
import torch

from common.common import logger
from utils.util import get_tokenizer
from models.model import SoftMaskedBertModel
from utils.util import data_inference_feature
from utils.util import processing_predict_result


def predict(data_file, pretrain_model_path, pretrain_model_type, model_path,
            max_length, hidden_size, output_file):
    # 读取数据
    target_sent = []
    source_sent = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            target, source, _ = line.strip().split('\t')
            target_sent.append(target)
            source_sent.append(source)
    device = "cpu"
    # tokenizer
    tokenizer = get_tokenizer(pretrain_model_path, pretrain_model_type)
    # model
    model = SoftMaskedBertModel(pretrain_model_path, hidden_size, pretrain_model_type,
                                tokenizer.mask_token_id, device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)),
                          strict=True)
    model.to(device)
    result = []
    for text in source_sent:
        # get text feature
        input_token, attention_mask, length, text_token = data_inference_feature(text, tokenizer, max_length)
        candidates = []
        with torch.no_grad():
            input_token = input_token.to(device)
            attention_mask = attention_mask.to(device)
            prob, logits = model(input_token, input_token, attention_mask)
            output_tensor = torch.nn.Softmax(dim=-1)(logits)
            pred_ = torch.argmax(output_tensor, dim=-1).tolist()
            prob = torch.round(prob)
            prob = torch.round(prob).squeeze() * attention_mask
            prob = prob.data.cpu().numpy().tolist()[0]
            # cls, sep 的位置为0
            prob[0] = 0
            prob[length - 1] = 0
            detector_pred_index = [i for i in range(len(prob)) if prob[i] == 1]
            # pred_
            pred_result = input_token.clone().data.cpu().numpy().tolist()[0]
            if detector_pred_index:
                for item in detector_pred_index:
                    pred_result[item] = pred_[0][item]
            pred_result = pred_result[:length][1:-1]
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_result)
            result.append(''.join(pred_tokens))

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(target_sent)):
            f.write(target_sent[i] + "\t" + result[i] + "\n")


if __name__ == "__main__":
    data_file = "./data/sighan_data_ori/test_data.txt"
    pretrain_model_path = "D:/Spyder/pretrain_model/transformers_torch_tf/bert-base-chinese/"
    pretrain_model_type = "bert"
    model_path = "./output/model.pt"
    max_length = 64
    hidden_size = 256
    output_file = "./result/predict_result.txt"
    predict(data_file, pretrain_model_path, pretrain_model_type, model_path,
            max_length, hidden_size, output_file)
