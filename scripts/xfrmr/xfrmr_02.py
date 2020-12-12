# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
# https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering/#data
import os
import platform
import sys
PATH = '/Users/dhanley/Documents/riiid/' \
    if platform.system() == 'Darwin' else '/mount/riiid'
os.chdir(PATH)
sys.path.append(PATH)
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
import platform
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import log_loss
from tools.utils import get_logger, SpatialDropout, split_tags
from tools.config import load_config


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)

logger = get_logger('Train', 'INFO')

device = 'cpu' if platform.system() == 'Darwin' else 'cuda'
CUT=0
DIR='val'
VERSION='V1'
debug = False
validaten_flg = False

FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', \
                       'timestamp', 'user_answer']
logger.info(f'Loaded columns {", ".join(FILTCOLS)}')

valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]
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
EXTRACOLS = ['lag_time_cat', 'elapsed_time_cat']
#self = SAKTDataset(train, MODCOLS, PADVALS)


class SAKTDataset(Dataset):
    def __init__(self, data, basedf, cols, padvals, extracols, 
                 maxseq = 100, has_target = True): 
        super(SAKTDataset, self).__init__()
        
        self.cols = cols
        self.extracols = extracols
        self.data = data
        self.data['base'] = 0
        if basedf is not None: 
            self.base = basedf
            self.base['base'] = 1
            self.data = pd.concat([self.base, self.data], 0)
        self.data = self.data.sort_values(['user_id', 'timestamp']).reset_index(drop = True)
        self.padvals = padvals
        self.uidx = self.data.reset_index()\
            .groupby(['user_id'])['index'].apply(list).to_dict()
        self.quidx = self.data.query('base==0').reset_index()[['user_id', 'index']].values
        
        if basedf is None:
            self.quidx = self.quidx[np.random.choice(self.quidx.shape[0], 10**6, replace=False)]

            
        self.dfmat = self.data[self.cols].values
        self.padmat = self.padvals[self.cols].values
        self.users = self.data.user_id.unique() 
        self.maxseq = maxseq
        self.has_target = has_target
        self.targetidx =  [self.cols.index(c) for c in \
                           ['answered_correctly', 'user_answer', 'correct_answer']]
        self.padtarget = np.array([self.padvals[self.targetidx].tolist()])
        self.yidx = self.cols.index('answered_correctly') 
        self.timecols = [self.cols.index(c) for c in ['timestamp','prior_question_elapsed_time']]
    
    def __len__(self):
        
        return len(self.quidx)
    
    def __getitem__(self, idx):
        
        # Get index of user and question
        u,q = self.quidx[idx]
        # Pull out ths user index sequence
        useqidx = self.uidx[u]
        # Pull out position of question
        cappos  = useqidx.index(q) + 1
        # Pull out the sequence of questions up to that question
        useqidx = useqidx[:cappos][-100:]
        umat = self.dfmat[useqidx].astype(np.float32)
        
        if umat.shape[0] < self.maxseq:
            padlen = self.maxseq - umat.shape[0] 
            upadmat = np.tile(self.padmat, (padlen, 1))
            umat = np.concatenate((upadmat, umat), 0)
            
        # convert time to lag
        umat[:, self.timecols[0]][1:] = umat[:, self.timecols[0]][1:] - umat[:, self.timecols[0]][:-1]
        umat[:, self.timecols[0]][0] = 0
        
        # Time embeddings
        timeemb = (umat[:, self.timecols] / 1000).clip(0, 300).astype(np.int16)
        umat = np.concatenate((umat, timeemb), 1)
        
        # preprocess continuous time - try log scale and roughly center it
        umat[:, self.timecols] = np.log10( 1.+ umat[:, self.timecols] / (60 * 1000) ) 
        
        if self.has_target:
            target = umat[-1, self.yidx ]
            umat[:, self.targetidx] = np.concatenate((self.padtarget, \
                                                      umat[:-1, self.targetidx]), 0)
        if target > 1:
            logger.info(f'{target}\t{u},{q}\t{idx}' )
        umat = torch.tensor(umat).float()
        target = torch.tensor(target)
        
        return umat, target
    
class LearnNet(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device = device, dropout = 0.2):
        super(LearnNet, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.padvals = padvals
        self.extracols = extracols
        self.modcols = modcols + extracols
        self.contcols = contcols
        self.embcols = ['content_id', 'part']
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.tag_wts = torch.ones((sum(self.tag_idx), 16))  / sum(self.tag_idx)
        self.tag_wts = nn.Parameter(self.tag_wts)
        self.tag_wts.requires_grad = True
        
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        LSTM_UNITS = 32 + 4 + 16 * 3 # + len(self.contcols)
        
        self.lstm1 = nn.LSTM(LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.bnlstm = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x):
        
        # Categroical embeddings
        embcat = torch.cat([
            self.emb_content_id(x[:,:, self.modcols.index('content_id')].long()),
            self.emb_part(x[:,:, self.modcols.index('part')].long()), 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(x[:,:, self.modcols.index('lag_time_cat')].long()), 
            self.emb_elapsed_time(x[:,:, self.modcols.index('elapsed_time_cat')].long())
            ], 2)
        embcat = self.embedding_dropout(embcat)
        
        ## Continuous
        # contmat  = x[:,:, self.cont_idx]
        # Weighted sum of tags - hopefully good weights are learnt
        # xinp = torch.cat([embcat, contmat], 2)
        xinp = embcat
        
        h_lstm1, _ = self.lstm1(xinp)
        # Take last hidden unit
        hidden = self.dropout(self.bnlstm(h_lstm1[:,-1,:]))
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out

logger.info('Create model and loaders')
model = self =  LearnNet(MODCOLS, CONTCOLS, PADVALS, EXTRACOLS)
model.to(device)

LR = 0.001
DECAY = 0.0
# Should we be stepping; all 0's first, then all 1's, then all 2,s 
trndataset = SAKTDataset(train, None, MODCOLS, PADVALS, EXTRACOLS)
valdataset = SAKTDataset(valid, train, MODCOLS, PADVALS, EXTRACOLS)
loaderargs = {'num_workers' : 8, 'batch_size' : 256}
trnloader = DataLoader(trndataset, shuffle=True, **loaderargs)
valloader = DataLoader(valdataset, shuffle=False, **loaderargs)
# x, y = next(iter(trnloader))

criterion =  nn.BCEWithLogitsLoss()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [ {'params': [p for n, p in param_optimizer] } ]
optimizer = torch.optim.Adam(plist, lr=LR)


if device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()


logger.info('Start training')
best_val_loss = 100.
trn_lossls = []
for epoch in range(50):
    for param in model.parameters():
        param.requires_grad = True
    model.train()  
    pbartrn = tqdm(enumerate(trnloader), 
                total = len(trndataset)//loaderargs['batch_size'], 
                desc=f"Train epoch {epoch}", ncols=0)
    trn_loss = 0.
    
    for step, batch in pbartrn:

        optimizer.zero_grad()
        x, y = batch
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = torch.autograd.Variable(y)
        
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        '''
        with autocast():
            out = model(x)
            loss = criterion(out, y)
        if device != 'cpu':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        '''
        
        
        trn_loss += loss.item()
        trn_lossls.append(loss.item())
        trn_lossls = trn_lossls[-1000:]
        pbartrn.set_postfix({'train loss': trn_loss / (step + 1), \
                             'last 1000': sum(trn_lossls) / len(trn_lossls) })
    
    pbarval = tqdm(enumerate(valloader), 
                total = len(valdataset)//loaderargs['batch_size'], 
                desc=f"Valid epoch {epoch}", ncols=0)
    y_predls = []
    y_act = valid['answered_correctly'].values
    model.eval()
    for step, batch in pbarval:
        x, y = batch
        x = x.to(device, dtype=torch.float)
        with torch.no_grad():
            out = model(x)
        y_predls.append(out.detach().cpu().numpy())
        
    y_pred = np.concatenate(y_predls)
    auc_score = roc_auc_score(y_act, y_pred )
    logger.info(f'Valid AUC Score {auc_score:.5f}')


















