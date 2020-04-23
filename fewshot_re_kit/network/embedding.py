"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from bert import pytorch_pretrained_bert
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import argparse
import collections
import logging
import json
import re
import ipdb
from collections import Iterable
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel






class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):


    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
   

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
 
        if tokens_b:

            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:

            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
        #ipdb.set_trace()
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features,tokens


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
 

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_list):
    examples = []
    unique_id = 0
    for ins in input_list:
            ins = ins.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", ins)
            if m is None:
                text_a = ins
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples



class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length=100, word_embedding_dim=768, dpos_embedding_dim=50, dmask_embedding_dim=50):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.dpos_embedding_dim = dpos_embedding_dim
        self.dmask_embedding_dim = dmask_embedding_dim
        #self.bert_token = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bert_embedding = BertModel.from_pretrained('bert-base-uncased')
        self.bert_token = BertTokenizer.from_pretrained('./models/bert-base-uncased-vocab.txt')
        self.bert_embedding = BertModel.from_pretrained('./models')
 
        
        # Word embedding
        #unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        #blk = torch.zeros(1, word_embedding_dim)
        #word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(400002, 50, padding_idx=word_vec_mat.shape[0] + 1)
        #self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))
        #self.bword_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))
        #Position Embedding
        self.pos1_embedding = nn.Embedding(80, 5, padding_idx=0)
        self.pos2_embedding = nn.Embedding(80, 5, padding_idx=0)
        self.dpos1_embedding = nn.Embedding(2*self.max_length,dpos_embedding_dim,padding_idx=0)
        self.dpos2_embedding = nn.Embedding(2*self.max_length,dpos_embedding_dim,padding_idx=0)
        #dmask embedding
        self.dmask1_embedding = nn.Embedding(2*self.max_length,dmask_embedding_dim,padding_idx=0)
        self.dmask2_embedding = nn.Embedding(2*self.max_length,dmask_embedding_dim,padding_idx=0)

    def len2mask(self, bert):
        bert = bert.sum(-1)
        mask = torch.ones(bert.size(0),bert.size(1)).type(torch.cuda.FloatTensor)
        mask = torch.where(bert==0,bert,mask)
        return mask

    def forward(self, inputs,sent):
        #word = inputs['word']
        #sent = inputs['sent']
        dpos1 = inputs['dpos1']
        dpos2 = inputs['dpos2']
        dmask1 = inputs['dmask1']
        dmask2 = inputs['dmask2']
        examples = read_examples(sent) 
        features,tokens = convert_examples_to_features(examples=examples, seq_length=self.max_length, tokenizer=self.bert_token)
        #ipdb.set_trace()
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature 
        #ipdb.set_trace()
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).cuda()
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long).cuda()
        all_encoder_layers, _ = self.bert_embedding(all_input_ids , token_type_ids=None, attention_mask=all_input_mask)
        bert = all_encoder_layers[-1]
      
        mask = self.len2mask(bert)
        dpos1_emb = self.dpos1_embedding(dpos1)
        dpos2_emb  = self.dpos2_embedding(dpos2)
        dmask1_emb = self.dmask1_embedding(dmask1)
        dmask2_emb = self.dmask2_embedding(dmask2)
        
        demb1 = torch.cat([dpos1_emb,dmask1_emb],2)
        demb2 = torch.cat([dpos2_emb,dmask2_emb],2)
        demb = torch.cat([dmask1_emb,dmask2_emb],2)
        #ipdb.set_trace()
        bert = torch.cat([bert,
                          dpos1_emb,
                          dpos2_emb],
                          2)
      
        #x = torch.cat([self.word_embedding(word),
                            #self.pos1_embedding(pos1), 
                            #self.pos2_embedding(pos2)],
                           #2)   #400 40 60
        #print(all_input_mask)
        return bert,demb,all_input_mask


