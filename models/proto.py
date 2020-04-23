import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#myfont = matplotlib.font_manager.FontProperties(fname='STSONG.TTF',size = 20)
#from pytorch_pretrained_bert import BertTokenizer

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.tanh = nn.Tanh()
        #self.fc2 = nn.Linear(460,230)
        #self.dfc = nn.Linear(200,hidden_size)
        #self.bert_token = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.pool = nn.MaxPool1d(100)        
        self.index = 0
        self.allre = 0
        self.correct = 0
        
    def __dist__(self, x, y, dim):
        #ipdb.set_trace()
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        #ipdb.set_trace()
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    '''
    def display_attention(self,target_class,query,attention,index,label,pred,entity1,entity2):
        ipdb.set_trace()
        class_dict = {'P177':'(brige-river)','P364':'(national)','P2094':'(person-weight)','P25':'(parents)','P26':'(couple)','P40':'(uncle)','P155':'(and)','P206':'(island)','P410':'(Military rank)','P361':'(Masterpiece)','P921':'(public)','P59':'(star)','P412':'(singer-range)','P463':'(member)','P413':'(role)','P641':'(athlete)'}
        displayx = []
        display_query = []
        display_query.append('[head]')
        for q in query:
          display_query.append(q)
        display_query.append('[tail]')
        
        fig, ax = plt.subplots(figsize=(len(query)+2,20))
        attention=attention[:,0:len(query)+2].cpu().detach().numpy()
        for i in range(0,len(query)+2):
          displayx.append(display_query[i] +'\n'+ str(round(attention[0][i],2)))
        cax=ax.matshow(attention,cmap='bone')
        #ipdb.set_trace()
        title = str(label+'--------'+pred+'=='+entity1+','+entity2)
        ax.set_title(title)
        tick1=np.arange(0,len(query)+2)
        ax.set_xticks(tick1)
        ax.set_xticklabels(displayx)  
        ax.xaxis.set_ticks_position('bottom')
        tick2=np.arange(1)
        ax.set_yticks(tick2)
      
        plt.title(title)
        plt.savefig('./wordatt26/att'+str(index))
        plt.close()
        print(index)
        #print('attention sum :', attention.sum())
    '''
    def forward(self, support, query, N, K, Q,label,target_class):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ''' 

             
        
        orisent_s = support['sent']
        sent_s = [x for z in orisent_s for y in z for x in y]
        #ipdb.set_trace()
        support,score= self.sentence_encoder(support,sent_s,N,K) 
        orisent_q = query['sent']
        sent_q = [x for z in orisent_q for x in z]
        query,score= self.sentence_encoder(query,sent_q,N,Q) # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query  = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)
        #ipdb.set_trace()
        #query1 = query
        #query2 = self.tanh(self.fc2(query))
        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
        D = support.size(-1)
        support = torch.mean(support, 2) # Calculate prototype for each class
        # Prototypical Networks 


        logits = -self.__batch_dist__(support, query)

        #logits1 = -self.__batch_dist__(support,query1)
        #logits2 = -self.__batch_dist__(support,query2)
        #_, pred = torch.max(logits1.view(-1, N), -1)
        mask = torch.zeros(label.size())
        
        
        if self.training:
            label2 = torch.clone(label)
            logits1 = logits
            logits2 = self.tanh(self.fc2(logits))
            _, pred = torch.max(logits1.view(-1, N), -1)
            # ipdb.set_trace()
            for i in range(0,label.size(1)):
               if label[0][i] != pred[i]:
                  label2[0][i] = pred[i]
                  mask[0][i] = 1
        else:
            #ipdb.set_trace()
            logits1 = logits
            logits2 = logits
            _, pred = torch.max(logits1.view(-1, N), -1)
            label2 = label
       
        mask = mask.unsqueeze(-1).cuda()
        #ipdb.set_trace()
        #logits = nn.softmax(logits)
        #logits2 = nn.softmax(logits2)
        
        '''
        if not self.training:          
          for i in range(0,label.size(1)):           
              if target_class[0][int(label[0][i])] == 'P25':
                self.allre += 1
              
                q_token = self.bert_token.tokenize(orisent_q[0][i])
                #ipdb.set_trace()
                indpos1 = torch.eq(dpos1[i],100) 
                indpos2 = torch.eq(dpos2[i],100)
                _, entindex1 = indpos1.max(dim=0)
                _, entindex2 = indpos2.max(dim=0)
                entity1 = q_token[entindex1]
                entity2 = q_token[entindex2]
              
                if label[0][i] == pred[i]:
                    self.correct += 1
          if self.allre:
             print (self.correct, self.allre , self.correct/self.allre)
  
                    #self.display_attention(target_class, q_token, score[i].unsqueeze(0),self.index,target_class[0][int(label[0][i])],target_class[0][int(pred[i])],entity1,entity2)
                    #self.index += 1
        '''
        #ipdb.set_trace()
        return logits1, pred ,label2,logits2,mask#,logits2#loss2#,logits2, label2
