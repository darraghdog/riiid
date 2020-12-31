# Ripped from https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/zoo/classifiers.py
from functools import partial
#import timm
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import log_loss
import pandas as pd
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn.functional as F
from torch import nn
#from pytorch_transformers.modeling_bert import BertConfig, BertEncoder
from transformers import XLMModel, XLMConfig

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

# inputs, lengths = hiddenq, m.sum(1)
class Attention21(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention21, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        weights = torch.zeros(1, hidden_size)
        weights[:, -1] = 1.
        self.att_weights = nn.Parameter(weights, requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, mask):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.tensor(mask.float(), requires_grad=True)

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()
        if representations.ndim == 1:
            representations = representations.unsqueeze(0)

        return representations, attentions
    
    

class LearnNet24(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device = 'cpu', dropout = 0.2, model_type = 'lstm', 
                 hidden = 512):
        super(LearnNet24, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        self.model_type = model_type
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_content_id_prior = nn.Embedding(13526*3, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_prior = nn.Embedding(3, 2)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 5)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        IN_UNITSQ = \
                2 * self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                2 * self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim + \
                        self.emb_prior.embedding_dim # + self.emb_content_id_prior.embedding_dim
        IN_UNITSQA = self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                len(self.contcols)
        LSTM_UNITS = hidden
        self.diffsize = self.emb_content_id.embedding_dim + self.emb_part.embedding_dim 
        
        #self.seqnet1 = nn.LSTM(IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        #self.seqnet2 = nn.LSTM(IN_UNITSQA + LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)

        self.seqnet = nn.LSTM(IN_UNITSQA + IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.atten2 = Attention21(LSTM_UNITS, batch_first=True) # 2 is bidrectional
        
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x, m = None):
        
        content_id_prior = x[:,:,self.modcols.index('content_id')] * 3 + \
                            x[:,:, self.modcols.index('prior_question_had_explanation')]
        embcatq = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            # self.emb_content_id_prior(  content_id_prior.long()  ),
            self.emb_prior( x[:,:, self.modcols.index('prior_question_had_explanation')].long() ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            #self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            #self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            #self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatqdiff = embcatq[:,:,:self.diffsize] - embcatq[:,-1,:self.diffsize].unsqueeze(1)
            
        # Categroical embeddings
        embcatqa = torch.cat([
            #self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            #self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            #self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            #self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            #self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            #(self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Weighted sum of tags - hopefully good weights are learnt
        '''
        xinpq = torch.cat([embcatq, embcatqdiff], 2)
        hiddenq, lengths = self.seqnet1(xinpq)
        xinpqa = torch.cat([embcatqa, contmat, hiddenq], 2)
        hiddenqa, _ = self.seqnet2(xinpqa)
        '''
        xinpqa = torch.cat([embcatqa, contmat, embcatq, embcatqdiff], 2)
        hiddenqa, _ = self.seqnet(xinpqa)
        hiddenqa, _ = self.atten2(hiddenqa, m)
        if hiddenqa.ndim == 1:
            hiddenqa = hiddenqa.unsqueeze(0)
        
        # Take last hidden unit
        hidden = hiddenqa#[:,-1,:]
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out

class LearnNet21(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device = 'cpu', dropout = 0.2, model_type = 'lstm', 
                 hidden = 512):
        super(LearnNet21, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        self.model_type = model_type
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_content_id_prior = nn.Embedding(13526*3, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_prior = nn.Embedding(3, 2)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 5)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        IN_UNITSQ = \
                2 * self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                2 * self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim + \
                        self.emb_prior.embedding_dim # + self.emb_content_id_prior.embedding_dim
        IN_UNITSQA = self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                len(self.contcols)
        LSTM_UNITS = hidden
        self.diffsize = self.emb_content_id.embedding_dim + self.emb_part.embedding_dim 
        
        self.seqnet1 = nn.LSTM(IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.seqnet2 = nn.LSTM(IN_UNITSQA + LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.atten2 = Attention21(LSTM_UNITS, batch_first=True) # 2 is bidrectional
        
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x, m = None):
        
        content_id_prior = x[:,:,self.modcols.index('content_id')] * 3 + \
                            x[:,:, self.modcols.index('prior_question_had_explanation')]
        embcatq = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            # self.emb_content_id_prior(  content_id_prior.long()  ),
            self.emb_prior( x[:,:, self.modcols.index('prior_question_had_explanation')].long() ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            #self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            #self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            #self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatqdiff = embcatq[:,:,:self.diffsize] - embcatq[:,-1,:self.diffsize].unsqueeze(1)
            
        # Categroical embeddings
        embcatqa = torch.cat([
            #self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            #self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            #self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            #self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            #self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            #(self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Weighted sum of tags - hopefully good weights are learnt
        xinpq = torch.cat([embcatq, embcatqdiff], 2)
        hiddenq, lengths = self.seqnet1(xinpq)
        xinpqa = torch.cat([embcatqa, contmat, hiddenq], 2)
        hiddenqa, _ = self.seqnet2(xinpqa)
        hiddenqa, _ = self.atten2(hiddenqa, m)
        if hiddenqa.ndim == 1:
            hiddenqa = hiddenqa.unsqueeze(0)
        
        # Take last hidden unit
        hidden = hiddenqa#[:,-1,:]
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out
    

class LearnNet20(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device, dropout = 0.2, model_type = 'lstm', 
                 hidden = 512):
        super(LearnNet20, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        self.model_type = model_type
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_content_id_prior = nn.Embedding(13526*3, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_prior = nn.Embedding(3, 2)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 5)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        IN_UNITSQ = \
                2 * self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                2 * self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim + \
                        self.emb_prior.embedding_dim + self.emb_content_id_prior.embedding_dim
        IN_UNITSQA = self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                len(self.contcols)
        LSTM_UNITS = hidden
        self.diffsize = self.emb_content_id.embedding_dim + self.emb_part.embedding_dim 
        
        self.seqnet1 = nn.LSTM(IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.seqnet2 = nn.LSTM(IN_UNITSQA + LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
            
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x, m = None):
        
        content_id_prior = x[:,:,self.modcols.index('content_id')] * 3 + \
                            x[:,:, self.modcols.index('prior_question_had_explanation')]
        embcatq = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            self.emb_content_id_prior(  content_id_prior.long()  ),
            self.emb_prior( x[:,:, self.modcols.index('prior_question_had_explanation')].long() ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            #self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            #self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            #self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatqdiff = embcatq[:,:,:self.diffsize] - embcatq[:,-1,:self.diffsize].unsqueeze(1)
            
        # Categroical embeddings
        embcatqa = torch.cat([
            #self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            #self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            #self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            #self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            #self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            #(self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Weighted sum of tags - hopefully good weights are learnt
        xinpq = torch.cat([embcatq, embcatqdiff], 2)
        hiddenq, _ = self.seqnet1(xinpq)
        xinpqa = torch.cat([embcatqa, contmat, hiddenq], 2)
        hiddenqa, _ = self.seqnet2(xinpqa)
        
        # Take last hidden unit
        hidden = hiddenqa[:,-1,:]
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out

class LearnNet12(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device, dropout, model_type, hidden):
        super(LearnNet12, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        self.model_type = model_type
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 5)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        IN_UNITSQ = \
                self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim
        IN_UNITSQA = self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                len(self.contcols)
        LSTM_UNITS = hidden
        
        self.seqnet1 = nn.LSTM(IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.seqnet2 = nn.LSTM(IN_UNITSQA + LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
            
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x, m = None):
        
        embcatq = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            #self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            #self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            #self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        
        # Categroical embeddings
        embcatqa = torch.cat([
            #self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            #self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            #self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            #self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            #self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            #(self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Weighted sum of tags - hopefully good weights are learnt
        xinpq = embcatq
        hiddenq, _ = self.seqnet1(xinpq)
        xinpqa = torch.cat([embcatqa, contmat, hiddenq], 2)
        hiddenqa, _ = self.seqnet2(xinpqa)
        
        # Take last hidden unit
        hidden = hiddenqa[:,-1,:]
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out

class LearnNet14(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device, dropout, model_type, hidden, maxseq):
        super(LearnNet14, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        self.model_type = model_type
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 8)
        self.pos_embedding = nn.Embedding(maxseq, 16)
            
        self.tag_idx = torch.tensor(['tag' == i[:-1] for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        IN_UNITSQ = \
                self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim + \
                        self.pos_embedding.embedding_dim
        IN_UNITSQA = self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                self.pos_embedding.embedding_dim + len(self.contcols)
        LSTM_UNITS = hidden
        
        self.xcfg = XLMConfig()
        self.xcfg.causal = True
        self.xcfg.emb_dim = IN_UNITSQ
        self.xcfg.max_position_embeddings = maxseq
        self.xcfg.n_layers = 2 #args.n_layers
        self.xcfg.n_heads = 8#args.n_heads
        self.xcfg.dropout = dropout
        self.xcfg.return_dict = False
        self.seqnetq  = XLMModel(self.xcfg)
        self.xcfgqa = self.xcfg
        self.xcfgqa.emb_dim = IN_UNITSQ + IN_UNITSQA 
        self.seqnetqa  = XLMModel(self.xcfgqa)
        
        #self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(IN_UNITSQ + IN_UNITSQA , 1)
        
    def forward(self, x, m = None, device = 'cuda'):
        
        # Position ID
        bsize, seqlen, dim = x.shape 
        pos_id = torch.arange(seqlen).repeat(bsize).reshape(bsize, seqlen).to(device)
        
        embcatq = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            self.emb_lpart(  x[:,:, self.modcols.index('lecture_part')].long()  ), 
            self.emb_ltag(  x[:,:, self.modcols.index('lecture_tag')].long()  ) , 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.pos_embedding( pos_id  ), 
            ], 2)
        
        # Categroical embeddings
        embcatqa = torch.cat([
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  ),
            self.pos_embedding( pos_id  ), 
            ], 2)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Transformer
        inputs = {'input_ids': None, 'inputs_embeds': embcatq, 'attention_mask': m}
        hidden = self.seqnetq(**inputs)[0]
        embcatqa = torch.cat([hidden, embcatqa, contmat], 2)
        inputs = {'input_ids': None, 'inputs_embeds': embcatqa, 'attention_mask': m}
        hidden = self.seqnetqa(**inputs)[0]
        
        # Take last hidden unit
        out = self.linear_out(hidden[:,-1,:]).flatten()
        
        return out