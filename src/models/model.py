# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 17:00
# @Author  : zxf
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import BertModel
from transformers import AutoTokenizer
from transformers import BertTokenizer


class SoftMaskedBertModel(nn.Module):

    def __init__(self, pretrain_model_path, hidden_size, pretrain_model_type,
                 mask_token_id, device):
        super(SoftMaskedBertModel, self).__init__()
        self.pretrain_model_path = pretrain_model_path
        self.hidden_size = hidden_size
        self.pretrain_model_type = pretrain_model_type
        self.mask_token_id = mask_token_id
        self.device = device
        if self.pretrain_model_path == "bert":
            self.mlm_model = BertModel.from_pretrained(self.pretrain_model_path)
        else:
            self.mlm_model = AutoModel.from_pretrained(self.pretrain_model_path)

        self.embedding_size = self.mlm_model.config.hidden_size
        self.embedding = self.mlm_model.embeddings
        self.vocab_size = self.mlm_model.config.vocab_size
        self.mlm_model_encoder = self.mlm_model.encoder
        self.mask_embedding = self.embedding(torch.tensor([[self.mask_token_id]],
                                                          dtype=torch.long)).detach().to(self.device)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, num_layers=2,
                          bidirectional=True,
                          batch_first=True)
        self.rnn_linear = nn.Linear(self.hidden_size * 2, 1)
        self.classifier = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, input_tokens, output_tokens, input_mask):
        # get bert embedding
        input_embedding = self.embedding(input_tokens).to(self.device)
        # rnn
        rnn_hidden, _ = self.rnn(input_embedding)
        rnn_hidden = self.rnn_linear(rnn_hidden)
        # 进行sigmoid
        rnn_hidden_prob = nn.Sigmoid()(rnn_hidden)
        # correct network
        corr_embedding = rnn_hidden_prob * self.mask_embedding + (1 - rnn_hidden_prob) * input_embedding
        # bert 12block encoder
        bert_mask = self.mlm_model.get_extended_attention_mask(input_mask,
                                                               output_tokens.shape,
                                                               input_tokens.device)
        bert_out = self.mlm_model_encoder(hidden_states=corr_embedding,
                                          attention_mask=bert_mask)
        # 残差
        output = bert_out['last_hidden_state'] + input_embedding
        logits = self.classifier(output)
        return rnn_hidden_prob, logits