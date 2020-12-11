# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
# https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering/#data
import os
os.chdir('/Users/dhanley/Documents/riiid/')
import sys
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import lightgbm as lgb
import warnings
from scipy import sparse
from scripts.utils import Iter_Valid, dumpobj, loadobj
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)

pd.set_option('display.width', 1000)

CUT=0
DIR='val'
VERSION='V1'
debug = False
validaten_flg = False

FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', \
                       'timestamp', 'user_answer']
valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS].head(10000)
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS].head(10**6)
gc.collect()

train = train.sort_values(['user_id', 'timestamp']).reset_index(drop = True)
valid = valid.sort_values(['user_id', 'timestamp']).reset_index(drop = True)

train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype(np.float32).fillna(2).astype(np.int8)
valid['prior_question_had_explanation'] = valid['prior_question_had_explanation'].astype(np.float32).fillna(2).astype(np.int8)
train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].fillna(0).astype(np.int32)
valid['prior_question_elapsed_time'] = valid['prior_question_elapsed_time'].fillna(0).astype(np.int32)


# For start off remove lectures
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# Joins questions
def split_tags(x) : 
    if x == '': return [188]*6
    return list(map(int, x.split(' ')))+[188]*(6-len(x.split(' '))) 
qdf = pd.read_csv('data/questions.csv')
qdf[[f'tag{i}' for i in range(6)]] =  qdf.tags.fillna('').apply(split_tags).tolist()

keepcols = ['question_id', 'part', 'bundle_id', 'correct_answer'] + [f'tag{i}' for i in range(6)]
train = pd.merge(train, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
formatcols =  ['question_id', 'part', 'bundle_id', 'correct_answer', 'user_answer']+ [f'tag{i}' for i in range(6)]
train[formatcols] = train[formatcols].fillna(0).astype(np.int16)
valid[formatcols] = valid[formatcols].fillna(0).astype(np.int16)

# Create index for loader
trnidx = train.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()
validx = valid.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()

train.reset_index()[['user_id', 'index']].values


MODCOLS = ['content_id', 'content_type_id', 'prior_question_elapsed_time', \
           'prior_question_had_explanation', 'task_container_id', \
            'timestamp', 'part', 'bundle_id', \
                'answered_correctly', 'user_answer', 'correct_answer'] \
            + [f'tag{i}' for i in range(6)]
NOPAD = ['prior_question_elapsed_time', 'prior_question_had_explanation', \
             'timestamp', 'content_type_id']
EMBCOLS = ['content_id', 'part', 'bundle_id'] + [f'tag{i}' for i in range(6)]
TARGETCOLS = [ 'user_answer', 'answered_correctly', 'correct_answer']
CONTCOLS = ['timestamp', 'prior_question_elapsed_time', 'prior_question_had_explanation']
PADVALS = train[MODCOLS].max(0) + 1
PADVALS[NOPAD] = 0


#self = SAKTDataset(train, MODCOLS, PADVALS)

class SAKTDataset(Dataset):
    def __init__(self, data, cols, padvals, 
                 maxseq = 100, has_target = True): 
        super(SAKTDataset, self).__init__()
        
        self.cols = cols
        self.data = data
        self.padvals = padvals
        self.uidx = self.data.reset_index()\
            .groupby(['user_id'])['index'].apply(list).to_dict()
        self.quidx = self.data.reset_index()[['user_id', 'index']].values
        self.dfmat = self.data[self.cols].values
        self.padmat = self.padvals[self.cols].values
        self.users = self.data.user_id.unique() 
        self.maxseq = maxseq
        self.has_target = has_target
        self.targetidx =  [self.cols.index(c) for c in \
                           ['answered_correctly', 'user_answer', 'correct_answer']]
        self.padtarget = np.array([self.padvals[self.targetidx].tolist()])
        self.timecols = [self.cols.index(c) for c in ['timestamp','prior_question_elapsed_time']]
    
    def __len__(self):
        
        return len(self.quidx)
    
    def __getitem__(self, idx):
        
        # Get index of user and question
        u,q = self.quidx[idx]
        # Pull out ths user index sequence
        useqidx = self.uidx[u]
        # Pull out position of question
        cappos  = useqidx.index(q)
        # Pull out the sequence of questions up to that question
        umat = self.dfmat[useqidx[:cappos]]
        
        if umat.shape[0] >= self.maxseq:
            umat = umat[:self.maxseq]
        else:
            padlen = self.maxseq - umat.shape[0] 
            upadmat = np.tile(self.padmat, (padlen, 1))
            umat = np.concatenate((upadmat, umat), 0)
            
        # convert time to lag
        umat[:, self.timecols[0]][1:] = umat[:, self.timecols[0]][1:] - umat[:, self.timecols[0]][:-1]
        umat[:, self.timecols[0]][0] = 0
        
        # preprocess continuous time - try log scale
        umat[self.timecols] = np.log( 1.+ umat[self.timecols] / 1000 )
        
        if self.has_target:
            target = umat[-1, self.targetidx ]
            umat[:, self.targetidx] = np.concatenate((self.padtarget, \
                                                      umat[:-1, self.targetidx]), 0)
        
        umat = torch.tensor(umat).long()
        target = torch.tensor(target)
        
        return umat, target

# Should we be stepping; all 0's first, then all 1's, then all 2,s 
trndataset = SAKTDataset(train, MODCOLS, PADVALS)
loaderargs = {'num_workers' : 16, 'batch_size' : 256}
trnloader = DataLoader(trndataset, shuffle=True, **loaderargs)



for t, (X, y) in tqdm(enumerate(trnloader)):
    if t> 100:
        break
    
    

embdims = OrderedDict([('content_id', 32), 
           ('part', 4), 
           ('tag', 16), 
           ('bundle_id', 32)] )
embmax = OrderedDict([('content_id', 13525), 
           ('part', 9), 
           ('tag', 189), 
           ('bundle_id', 13525)] )
padvals = train.max(0) + 1
emb = dict((k, nn.Embedding(embmax[k]+1, dim)) for k,dim in embdims.items())
tag_idx = ['tag' in i for i in MODCOLS]
tag_wts = torch.ones((sum(tag_idx), 16), requires_grad=True) 
cont_idx = [MODCOLS.index(c) for c in CONTCOLS]

# Categorical embeddings
EMBCOLS1 = ['content_id', 'part'] #,'bundle_id']
embcat = torch.cat([emb[c](X[:,:, MODCOLS.index(c)]) for c in EMBCOLS1 ], 2)
embtag = (emb['tag'](X[:,:, tag_idx]) * tag_wts).sum(2)
contmat  = X[:,:, cont_idx]
# Weighted sum of tags - hopefully good weights are learnt
x = torch.cat([embcat, embtag, contmat], 2)















