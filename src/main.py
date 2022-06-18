# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 12:14
# @Author  : zxf
import os
import json
import traceback

import torch
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import getConfig
from common.common import logger
from utils.util import collate_fn
from utils.util import get_tokenizer
from utils.util import model_evaluate
from utils.util import model_evaluate_v2
from models.model import SoftMaskedBertModel
from utils.util import SoftMaskedBertDataset
from utils.util import get_random_wrong_data
from utils.util import model_evaluate_acc_prf


def main(config_file):
    Config = getConfig.get_config(config_file)
    os.environ["CUDA_VISIBLE_DEVICES"] = Config["gpu_ids"]
    device = "cpu" if Config["gpu_ids"] == "-1" else "cuda"

    if not os.path.exists(Config["output_path"]):
        os.mkdir(Config["output_path"])

    # read data
    # train_data = pd.read_csv(Config["train_data_path"])
    # test_data = pd.read_csv(Config["test_data_path"])
    train_data = pd.read_excel(Config["train_data_path"])
    test_data = pd.read_excel(Config["test_data_path"])
    logger.info("train data size:{}".format(train_data.shape))
    logger.info("test data size:{}".format(test_data.shape))

    # get tokenizer
    tokenizer = get_tokenizer(Config["pretrain_model_path"],
                              Config["pretrain_model_type"])
    # add mask data
    if Config["data_mask"]:
        train_ori_data = train_data["origin_text"].values.tolist()
        train_mask_data = get_random_wrong_data(train_ori_data, tokenizer,
                                                Config["mask_rate"])
        train_mask_df = pd.DataFrame(train_mask_data, columns=train_data.columns)
        train_data = pd.concat([train_data, train_mask_df], axis=0)
        # shuffle
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        logger.info("data mask")
        logger.info("train data shape:{}".format(train_data.shape))
    train_dataset = SoftMaskedBertDataset(train_data, tokenizer,
                                          Config["max_length"])
    dev_dataset = SoftMaskedBertDataset(test_data, tokenizer,
                                        Config["max_length"])
    train_dataloader = DataLoader(train_dataset, batch_size=Config["batch_size"],
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1,
                                collate_fn=collate_fn)
    logger.info("pre epoch data batch number:{}".format(len(train_dataloader)))
    # model
    model = SoftMaskedBertModel(Config["pretrain_model_path"],
                                Config["hidden_size"],
                                Config["pretrain_model_type"],
                                tokenizer.mask_token_id, device)
    model.to(device)

    optimizer = AdamW(params=model.parameters(), lr=Config["lr"])
    detector_loss = nn.BCELoss()
    corrector_loss = nn.CrossEntropyLoss()
    logger.info("model start training")
    best_acc = 0.0
    best_abs_acc = 0.0
    best_total_abs_acc = 0.0
    for epoch in range(Config["epoch"]):
        model.train()
        for step, batch_data in enumerate(train_dataloader):
            batch_input = batch_data["error_text"].to(device)
            batch_output = batch_data["correct_text"].to(device)
            batch_label = batch_data["error_label"].to(device)
            batch_mask = batch_data["mask"].to(device)
            detector_logits, corrector_logits = model(batch_input,
                                                      batch_output, batch_mask)
            corrector_logits = corrector_logits.permute(0, 2, 1)
            detector_logits = detector_logits.squeeze(-1)
            det_loss = detector_loss(detector_logits, batch_label)
            corr_loss = corrector_loss(corrector_logits, batch_output)
            loss = Config["gama"] * corr_loss + (1 - Config["gama"]) * det_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True) # retain_graph=True
            optimizer.step()
            if (step + 1) % Config["pre_epoch_print"] == 0:
                logger.info("epoch:{}/{} step:{}/{} loss:{}".format(epoch + 1,
                                                                    Config["epoch"],
                                                                    step + 1,
                                                                    len(train_dataloader),
                                                                    loss))
        # 模型验证
        # acc, abs_total_acc, word_acc, detector_acc = model_evaluate(dev_dataloader, model, device)
        # logger.info("acc:{}, abs_total_acc:{}, word_acc:{}, detector_acc:{}".format(acc, abs_total_acc,
        #                                                                             word_acc, detector_acc))
        # # word_acc, detector_acc, acc = model_evaluate_v2(dev_dataloader, model, device)
        # # logger.info("abs acc:{}, word_acc:{}, detector_acc:{}".format(acc, word_acc, detector_acc))
        # if acc >= best_abs_acc:
        #     best_abs_acc = acc
        # if abs_total_acc > best_total_abs_acc:
        #     best_total_abs_acc = abs_total_acc
        detector_acc, acc, precision, recall, f1 = model_evaluate_acc_prf(model, dev_dataloader, device)
        logger.info("detector_acc :{}, sent abs acc:{}, precision:{}, "
                    "recall:{} f1 score:{}".format(detector_acc, acc, precision, recall, f1))
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(Config["output_path"],
                                                        Config["model_name"]))
            logger.info("best acc: {}".format(best_acc))
    logger.info("best abs acc :{}".format(best_acc))
    # logger.info("best acc:{}, best abs acc:{}, best total abs acc:{}".format(best_acc,
    #                                                                          best_abs_acc,
    #                                                                          best_total_abs_acc))


if __name__ == "__main__":
    config_file = "./config/config.ini"
    main(config_file)