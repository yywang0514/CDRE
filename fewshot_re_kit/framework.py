import os
import json
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import ipdb
class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    
class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.KL = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')
        self.fc = nn.Linear(10,10)
        #self.softmax = F.softmax()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        if int(torch.__version__.split('.')[1]) < 4:
            return x.item()
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              ckpt_dir='./checkpoint',
              test_result_dir='./test_result',
              learning_rate=1e-5,
              lr_step_size=20000,
              weight_decay=1e-6,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              cuda=True,
              pretrain_model=None,
              optimizer=optim.Adam,
              noise_rate=0):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()
        model.train()
        print (model)
        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        #ipdb.set_trace()
        if len(sys.argv) > 5:
            use_bert = sys.argv[5]
        else :
            use_bert = None
        #if use_bert:
            #support, query, label = self.train_data_loader.next_batch(B, N_for_train, K, Q, noise_rate=noise_rate)
        #else:
            #support, query, label = self.train_data_loader.next_batch_nobert(B, N_for_train, K, Q, noise_rate=noise_rate)
        for it in range(start_iter, start_iter + train_iter):
            scheduler.step() 
            
            if use_bert:
                support, query, label, target_class= self.train_data_loader.next_batch(B, N_for_train, K, Q, noise_rate=noise_rate)
            else:
                support, query, label,target_class = self.train_data_loader.next_batch_nobert(B, N_for_train, K, Q, noise_rate=noise_rate)
            
            
            
            #ipdb.set_trace()
            logits1, pred,label2,logits2,mask= model(support, query, N_for_train, K, Q,label,target_class)
            #ipdb.set_trace()
            loss1 = model.loss(logits1,label)
            loss2 = model.loss(logits2,label2)
            p_r = F.softmax(logits1,dim=-1) * mask
            p_f = F.softmax(logits2,dim=-1) * mask
            #JS = 1/2*self.KL(p_r,(p_r+p_f)/2) + 1/2*self.KL(p_f,(p_f+p_r)/2)
            loss = loss1+loss2-self.KL(p_f,p_r)
            #print label
            #print pred
            #print '---------------------------------------'
            #print label
            #print '========================================'
            right = model.accuracy(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, noise_rate=noise_rate)
                model.train()
                if acc > best_acc: 
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name+'all3' + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc
                
        print("\n####################\n")
        print("Finish training " + model_name)
        test_acc = self.eval(model, B, N_for_eval, K, Q, test_iter, ckpt=os.path.join(ckpt_dir, model_name+'all3' + '.pth.tar'), noise_rate=noise_rate)
        print("Test accuracy: {}".format(test_acc))

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None,
            noise_rate=0): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        if len(sys.argv) > 5:
            use_bert = sys.argv[5]
        else :
            use_bert = None
        for it in range(eval_iter):
            #ipdb.set_trace()
            if use_bert:
              #ipdb.set_trace()
              support, query, label,target_class = eval_dataset.next_batch(B, N, K, Q, noise_rate=noise_rate)
            else:
              support, query, label,target_class = eval_dataset.next_batch_nobert(B, N, K, Q, noise_rate=noise_rate)
            
            
            #fp2 = open("index2relation","a")
            #for i, class_name in enumerate(target_classes):
                #print >>fp2, str(i) + ':'+ str(class_name)
                #print >>fp2, "=============================================="
            #fp2.close
            
            
            logits1, pred,label2,logits2,mask= model(support, query, N, K, Q, label,target_class)

            #logits, pred= model(support, query, N, K, Q)
            #=========================================================================================
            
            '''
            id2word = {v : k for k, v in word2id.items()}
            
            #label_select = label.view(-1)
            support_ = support['word'].view(B,N,K,-1)
            query_ = query['word'].view(B,N,Q,-1)
            label_ = label.view(B,N,Q)
            pred_ = pred.view(B,N,Q)
            
            
            sent_s =[]
            sent_q =[]
            for b_n in range(0,B):
              sent_batch_s =[]
              
              for c_n in range(0,N):
                sent_relation_s = []
                for k_n in range(0,K):
                  sent_one_s = []
                  for i in range(0,support_.size(3)):
                     if id2word[int(support_[b_n,c_n,k_n,i])] != 'BLANK':
                        sent_one_s.append(id2word[int(support_[b_n,c_n,k_n,i])].encode('unicode-escape').decode('string_escape'))
                  sent_relation_s.append(sent_one_s)                    
                sent_batch_s.append(sent_relation_s)  
              sent_s.append(sent_batch_s)
                    
            for b_n in range(0,B):
              sent_batch_q =[]
              for c_n in range(0,N):
                sent_relation_q = []
                for q_n in range(0,Q):
                  sent_one_q = []
                  for j in range(0,query_.size(3)):
                     if id2word[int(query_[b_n,c_n,q_n,j])] != 'BLANK':
                        sent_one_q.append(id2word[int(query_[b_n,c_n,q_n,j])].encode('unicode-escape').decode('string_escape'))
                  sent_relation_q.append(sent_one_q)
                sent_batch_q.append(sent_relation_q)
              sent_q.append(sent_batch_q)             
              
            #
            for b_n in range(0,B):
              for c_n in range(0,N):
                for q_n in range(0,Q):
                  if pred_[b_n,c_n,q_n] != label_[b_n,c_n,q_n]:
                     #relation_t = target_classes[label_[b_n,c_n,q_n]]
                     #relation_f = target_classes[pred_[b_n,c_n,q_n]]
                     wrong_q = sent_q[b_n][c_n][q_n]
                     s_true1 = sent_s[b_n][label_[b_n,c_n,q_n]][0]
                     s_true2 = sent_s[b_n][label_[b_n,c_n,q_n]][1]
                     s_false1 = sent_s[b_n][pred_[b_n,c_n,q_n]][0]
                     s_false2 = sent_s[b_n][pred_[b_n,c_n,q_n]][1]
                     fp = open("correct_q","a")
                     print >> fp, "step:"+ str(it)
                     print >> fp, target_classes[b_n]
                     print >> fp, "wrong query: "
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, wrong_q
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, "true relation: "+ target_classes[b_n][label_[b_n,c_n,q_n]]
                     print >> fp, "support of relation"+ target_classes[b_n][label_[b_n,c_n,q_n]] +":"
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, s_true1
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, s_true2
                     print >> fp, "----------------------------------------------------------------------"     
                     print >> fp, "false relation:" + target_classes[b_n][pred_[b_n,c_n,q_n]]
                     print >> fp, "support of relation"+ target_classes[b_n][pred_[b_n,c_n,q_n]]+":"
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, s_false1
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, s_false2
                     print >> fp, "----------------------------------------------------------------------"
                     print >> fp, "=========================================================================================="
                     fp.close()
            '''
            right = model.accuracy(pred, label)
            iter_right += self.item(right.data)
            iter_sample += 1
            #fp = open("38","a")
            #fp.write(('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r'))
            #fp.close()
            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()
        print("")
        return iter_right / iter_sample
