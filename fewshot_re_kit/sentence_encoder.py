import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from . import network
import sys
import ipdb
sys.path.append('..')
import fewshot_re_kit
from torch import autograd, optim, nn
from torch.autograd import Variable

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length=100, word_embedding_dim=768, pos_embedding_dim=50 ,dmask_embedding_dim=50,tag_embedding_dim=5,hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim,dmask_embedding_dim)
        self.encoder = network.encoder.Encoder(word_vec_mat,max_length, word_embedding_dim, pos_embedding_dim,dmask_embedding_dim, hidden_size)
        

    def forward(self, inputs,sent,N,K):
        
        x = self.embedding(inputs,sent)
        x = self.encoder(x)
        return x#,mask

class PCNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length=100, word_embedding_dim=50, pos_embedding_dim=5, dmask_embedding_dim=50,hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        #self.f1 = nn.Linear(230, 1)
        #self.f2 = nn.Linear(64, 32)
        #self.f3 = nn.Linear(64, 2)
       # self.f4 = nn.Linear(1,230)
        #self.drop = nn.Dropout(0.5)
        #self.conv = nn.Conv1d(60, 230, 3, padding=1)
        #self.pool = nn.MaxPool1d(max_length)
        #self.cost = nn.MSELoss(size_average=False) 
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim,dmask_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, dmask_embedding_dim,hidden_size)

    def forward(self, inputs,demb,sent,N,K): 
        '''
        dmask1 = inputs['dmask1']
        dmask2 = inputs['dmask2']
        midmask = inputs['mask']
        midmask = torch.where(midmask != 2 , torch.full_like(midmask, 0), midmask)
        
        dmask = midmask
        mask = torch.zeros(dmask1.size(0),dmask1.size(1))#.type(torch.cuda.FloatTensor)    
        mask = torch.where(dmask > 0 , torch.full_like(dmask, 1), dmask) 
        mask = mask.type(torch.cuda.FloatTensor).unsqueeze(-1)
        
        bert = inputs['bert']
        dpos1 = inputs['dpos1']
        dpos2 = inputs['dpos2']
        #__import__("ipdb").set_trace()
      
        mask = bert.sum(-1).type(torch.cuda.FloatTensor)            
        one = torch.ones(mask.size(0),mask.size(1)).cuda()
        zeros = torch.zeros(mask.size(0),mask.size(1)).cuda()       
        mask = torch.where(mask==0,zeros,one)#.unsqueeze(2)
 
      
        #__import__("ipdb").set_trace()
        seg_mask = inputs['bseg']
        '''
        x,demb= self.embedding(inputs,sent)
        #demb1 = (demb1 * dpos1_mask).sum(1)
        #demb2 = (demb2 * dpos2_mask).sum(1)
        x= self.encoder.pcnn(x)
        #ipdb.set_trace()
        return x#,demb1,demb2,x_nopool
      
      
      
class TransformerSentenceEncoder(nn.Module):
  
    def __init__(self,word_vec_mat, max_length=100, word_embedding_dim=768, dpos_embedding_dim=50,dmask_embedding_dim=50,hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, dpos_embedding_dim,dmask_embedding_dim)
        self.encoder1 = network.encoder.TransformerEncoder(max_length,word_embedding_dim + 2*dpos_embedding_dim,hidden_size)
        self.encoder2 = network.encoder.TransformerEncoder(max_length,2*dmask_embedding_dim,hidden_size)
        #self.encoder2 = GCNSentenceEncoder(word_vec_mat,max_length=100, word_embedding_dim=50, pos_embedding_dim=0,dmask_embedding_dim=50,hidden_size=230)
        self.fc = nn.Linear(self.hidden_size,1)
        self.dropout=nn.Dropout(0.5)
        #self.conv = nn.Conv1d(868, self.hidden_size, 3, padding=1)
        self.bpool = nn.MaxPool1d(max_length)
    def len2mask(self, bert):
        bert = bert.sum(-1)
        mask = torch.ones(bert.size(0),bert.size(1)).type(torch.cuda.FloatTensor)
        mask = torch.where(bert==0,bert,mask)
        return mask 
     
    def forward(self, inputs,sent,N,K):
        #ipdb.set_trace()
       # bert = inputs['bert']        
        #mask = self.len2mask(bert) 
        #dmask1 = inputs['dmask1']
        #dmask2 = inputs['dmask2']
        #midmask = inputs['mask']
        #midmask = torch.where(midmask != 2 , torch.full_like(midmask, 0), midmask)
        #dmask = midmask|dmask1|dmask2
        #mask = torch.zeros(dmask1.size(0),dmask1.size(1))#.type(torch.cuda.FloatTensor)    
        #mask = torch.where(dmask > 0 , torch.full_like(dmask, 1), dmask) 
        #mask = mask.type(torch.cuda.FloatTensor)#.unsqueeze(-1)
        x_emb,den_emb,mask = self.embedding(inputs,sent)   
        #ipdb.set_trace()
        #x_emb = self.dropout(x_emb)
       # mask = self.len2mask(bert) 
        #mask = inputs['mask']
        #x_cnn = self.conv(x_emb.transpose(1, 2))
        x_tran = self.encoder1(x_emb,mask)
        den_tran = self.encoder2(den_emb,mask)
        #ipdb.set_trace()
        tran_att = torch.sigmoid((self.fc(den_tran)))
        x = x_tran.transpose(1,2)*tran_att.transpose(1,2)
        x = self.bpool(x)
        
        #ipdb.set_trace()
        return x,tran_att
