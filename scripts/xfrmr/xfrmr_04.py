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
import argparse
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
from transformers import XLMModel, XLMConfig

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)

logger = get_logger('Train', 'INFO')

# funcs for user stats with loop
def add_user_feats(df, pdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
    cu = np.zeros(len(df), dtype=np.uint32)
    acsb = np.zeros(len(df), dtype=np.uint32)
    cb = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    #expacsu = np.zeros(len(df), dtype=np.uint32)
    #expcu = np.zeros(len(df), dtype=np.uint32)
    #pexpm = np.zeros((len(df), 1), dtype=np.uint8)
    contid = np.zeros((len(df), 1), dtype=np.uint8)
    qamat = np.zeros((len(df),2), dtype=np.float16)
    #partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
    #                                  'cu' : np.zeros(len(df), dtype=np.uint32)})

    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', \
                        'task_container_id', 'timestamp', 'content_type_id', 'user_answer']
    if not update:
        itercols = [f for f in itercols if f!='answered_correctly']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, eltim, cid, tcid, tstmp, ctype, ua = row
        else:
            u, pexp, eltim, cid, tcid, tstmp, ctype, ua = row
        
        if ctype==1:
            continue
        
        
        bid = bdict[cid]
        newbid = bid == pdicts['track_b'][u]
                                 
        ucid = f'{u}__{cid}'
        acsu[cnt] = pdicts['answered_correctly_sum_u_dict'][u]                        
        cu[cnt] = pdicts['count_u_dict'][u]
        acsb[cnt] = pdicts['answered_correctly_sum_b_dict'][u]
        cb[cnt] = pdicts['count_b_dict'][u]
        cidacsu[cnt] = pdicts['content_id_answered_correctly_sum_u_dict'][ucid]
        cidcu[cnt] = pdicts['content_id_count_u_dict'][ucid]
        #expacsu[cnt] = pdicts['pexp_answered_correctly_sum_u_dict'][u]
        #expcu[cnt] = pdicts['pexp_count_u_dict'][u]
        qamat[cnt] = pdicts['qaRatiocum'][u] / (pdicts['count_u_dict'][u] + 0.01), \
                pdicts['qaRatiocum'][u] / (pdicts['qaRatioCorrectcum'][u] + 0.01)
                
        #partsdict[part]['acsu'][cnt]  = pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        #partsdict[part]['cu'][cnt] = pdicts[f'{part}p_count_u_dict'][u]
        
        #for p in range(1,8):
        #    if p == part: continue
        #    partsdict[p]['cu'][cnt] = pdicts[f'{p}p_count_u_dict'][u]
        #    partsdict[p]['acsu'][cnt]  = pdicts[f'{p}p_answered_correctly_sum_u_dict'][u]

        if update:
            pdicts['count_u_dict'][u] += 1
            try:
                pdicts['qaRatioCorrectcum'][u] += pdicts['qaRatio'][(cid, pdicts['qaCorrect'][cid])]
            except:
                pdicts['qaRatioCorrectcum'][u] += 1.
            try:
                pdicts['qaRatiocum'][u] += pdicts['qaRatio'][(cid, ua)]
            except:
                pdicts['qaRatiocum'][u] += 0.1
                    
            pdicts['count_c_dict'][cid] += 1
            #pdicts[f'{part}p_count_u_dict'][u] += 1
            pdicts['content_id_count_u_dict'][ucid] += 1
            pdicts['count_b_dict'][u] = 1 if newbid else pdicts['count_b_dict'][u] + 1
            if newbid : pdicts['answered_correctly_sum_b_dict'][u] = 0
            if yprev: 
                pdicts['answered_correctly_sum_u_dict'][u] += 1
                pdicts['answered_correctly_sum_c_dict'][cid] += 1
                pdicts['answered_correctly_sum_b_dict'][u] += 1
                pdicts['content_id_answered_correctly_sum_u_dict'][ucid] += 1

                #pdicts[f'{part}p_answered_correctly_sum_u_dict'][u] += 1
            #if pexp:
            #    if yprev: 
            #        pdicts['pexp_answered_correctly_sum_u_dict'][u] += yprev
            #    pdicts['pexp_count_u_dict'][u] += 1
            pdicts['track_b'][u] = bid
                
    for t1, (matcu, matascu) in enumerate(zip([cu, cidcu, cb], [acsu,  cidacsu, acsb])):
        df[f'counts___feat{t1}'] = matcu
        df[f'avgcorrect___feat{t1}'] =  (matascu / (matcu + 0.001)).astype(np.float16)
        #gc.collect()
    df['cid_answered_correctly'] = acsu
    df[[f'rank_stats_{i}' for i in range(2)]] = qamat
    
    #for t, i in enumerate(range(1,8)):  
    #    df[f'counts___feat{t1+t+1}'] = partsdict[i]['cu']
    #    df[f'avgcorrect___feat{t1+t+1}'] =  (partsdict[i]['acsu']  / (partsdict[i]['cu'] + 0.001)).astype(np.float16)
    #    del partsdict[i]
    
    return df

DECAY = 0.0
logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--workers', type=int, default=8, help='number of cpu threads to use')
arg('--batchsize', type=int, default=1024)
arg('--lr', type=float, default=0.001)
arg('--epochs', type=int, default=20)
arg('--maxseq', type=int, default=128)
arg('--hidden', type=int, default=256)
arg('--n_layers', type=int, default=2)
arg('--n_heads', type=int, default=8)
arg('--model', type=str, default='lstm')
arg('--label-smoothing', type=float, default=0.01)
args = parser.parse_args()
logger.info(args)

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

# Joins questions
ldf = pd.read_csv('data/lectures.csv')
ldf.type_of = ldf.type_of.str.replace(' ', '_')
ldict = ldf.set_index('lecture_id').to_dict()
lecture_types = [t for t in ldf.type_of.unique() if t!= 'starter']


qdf = pd.read_csv('data/questions.csv')
qdf[[f'tag{i}' for i in range(6)]] =  qdf.tags.fillna('').apply(split_tags).tolist()

bdict = qdf.set_index('question_id')['bundle_id'].to_dict()
keepcols = ['question_id', 'part', 'bundle_id', 'correct_answer'] + [f'tag{i}' for i in range(6)]
train = pd.merge(train, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
formatcols =  ['question_id', 'part', 'bundle_id', 'correct_answer', 'user_answer']+ [f'tag{i}' for i in range(6)]
train[formatcols] = train[formatcols].fillna(0).astype(np.int16)
valid[formatcols] = valid[formatcols].fillna(0).astype(np.int16)

# How correct is the answer
def qaRanks(df):
    aggdf1 = df.groupby(['question_id', 'user_answer', 'correct_answer'])['answered_correctly'].count()
    aggdf2 = df.groupby(['question_id'])['answered_correctly'].count()
    aggdf = pd.merge(aggdf1, aggdf2, left_index=True, right_index = True).reset_index()
    aggdf.columns = ['question_id', 'user_answer', 'correct_answer', 'answcount', 'quescount']
    aggdf['answerratio'] = (aggdf.answcount / aggdf.quescount).astype(np.float16)
    rankDf = aggdf.set_index('question_id')[[ 'answerratio', 'answcount']].reset_index()
    qaRatio = aggdf.set_index(['question_id', 'user_answer']).answerratio.to_dict()
    qaCorrect = qdf.set_index('question_id').correct_answer.to_dict()
    return qaRatio, qaCorrect, rankDf

ix = train.content_type_id == False
qaRatio, qaCorrect, rankDf = qaRanks(train[ix])


train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype(np.float32).fillna(2).astype(np.int8)
valid['prior_question_had_explanation'] = valid['prior_question_had_explanation'].astype(np.float32).fillna(2).astype(np.int8)
train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].fillna(0).astype(np.int32)
valid['prior_question_elapsed_time'] = valid['prior_question_elapsed_time'].fillna(0).astype(np.int32)

content_df1 = train.query('content_type_id == 0')[['content_id','answered_correctly']]\
                .groupby(['content_id']).agg(['mean', 'count']).astype(np.float16).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c', 'answered_correctly_ct_c']
content_df2 = train.query('content_type_id == 0') \
                .groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float16).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df.columns
del content_df1, content_df2
gc.collect()

content_df.iloc[:,1:] = content_df.iloc[:,1:].astype(np.float16)
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")


# user stats features with loops
pdicts = {'answered_correctly_sum_u_dict' : defaultdict(int),
          'count_u_dict' : defaultdict(int),
          'answered_correctly_sum_b_dict' : defaultdict(int),
          'count_b_dict' : defaultdict(int),
          'answered_correctly_sum_c_dict' : defaultdict(int),
          'count_c_dict' : defaultdict(int),
          'track_b' : defaultdict(int),
          'content_id_answered_correctly_sum_u_dict' : defaultdict(int),
          'content_id_count_u_dict': defaultdict(int),
          'userAvgRatioCum': defaultdict(float),
          'userRatioCum' : defaultdict(int),
          
          'qaRatio' : qaRatio,
          'qaCorrect': qaCorrect,
          'qaRatiocum' : defaultdict(int),
          'qaRatioCorrectcum' : defaultdict(int),
          
          'content_id_lag' : defaultdict(int), 
          'pexp_answered_correctly_sum_u_dict' : defaultdict(int),
          'pexp_count_u_dict': defaultdict(int)}

#for p in train.part.unique():
#    pdicts[f'{p}p_answered_correctly_sum_u_dict'] =  defaultdict(int)
#    pdicts[f'{p}p_count_u_dict'] =  defaultdict(int)
    
train = add_user_feats(train, pdicts)
valid = add_user_feats(valid, pdicts)

# For start off remove lectures
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

train['content_user_answer']  = train['user_answer'] + 4 * train['content_id'].astype(np.int32)
valid['content_user_answer']  = valid['user_answer'] + 4 * valid['content_id'].astype(np.int32)
train['answered_correctly_ct_log'] = np.log1p(train['answered_correctly_ct_c'].fillna(0).astype(np.float32)) - 7.5
valid['answered_correctly_ct_log'] = np.log1p(valid['answered_correctly_ct_c'].fillna(0).astype(np.float32)) - 7.5


NORMCOLS = ['counts___feat0', 'counts___feat1', 'counts___feat2', 
            'cid_answered_correctly']
meanvals = np.log1p(train[NORMCOLS].fillna(0).astype(np.float32)).mean().values
train[NORMCOLS] = np.log1p(train[NORMCOLS].fillna(0).astype(np.float32)) - meanvals
valid[NORMCOLS] = np.log1p(valid[NORMCOLS].fillna(0).astype(np.float32)) - meanvals


# Create index for loader
trnidx = train.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()
validx = valid.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()

FEATCOLS = ['counts___feat0', 'avgcorrect___feat0', 'counts___feat1', 'avgcorrect___feat1', 
    'counts___feat2', 'avgcorrect___feat2', 'cid_answered_correctly', 
    'rank_stats_0', 'rank_stats_1']
MODCOLS = ['content_id', 'content_type_id', 'prior_question_elapsed_time', \
           'prior_question_had_explanation', 'task_container_id', \
            'timestamp', 'part', 'bundle_id', \
                'answered_correctly', 'user_answer', 'correct_answer', 'content_user_answer', 
                'answered_correctly_avg_c', 'answered_correctly_ct_log', 'attempts_avg_c'] \
            + [f'tag{i}' for i in range(6)] + FEATCOLS
EMBCOLS = ['content_id', 'part', 'bundle_id'] + [f'tag{i}' for i in range(6)]
TARGETCOLS = [ 'user_answer', 'answered_correctly', 'correct_answer', 'content_user_answer']

CONTCOLS = ['timestamp', 'prior_question_elapsed_time', 'prior_question_had_explanation', \
            'answered_correctly_avg_c', 'answered_correctly_ct_log', 'attempts_avg_c'] + FEATCOLS
NOPAD = ['prior_question_elapsed_time', 'prior_question_had_explanation', \
             'timestamp', 'content_type_id'] + CONTCOLS
PADVALS = train[MODCOLS].max(0) + 1
PADVALS[NOPAD] = 0
EXTRACOLS = ['lag_time_cat',  'elapsed_time_cat']
#self = SAKTDataset(train, MODCOLS, PADVALS)


class SAKTDataset(Dataset):
    def __init__(self, data, basedf, cols, padvals, extracols, 
                 maxseq = args.maxseq, has_target = True): 
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
        
        #if basedf is None:
        #    self.quidx = self.quidx[np.random.choice(self.quidx.shape[0], 2*10**6, replace=False)]
        
        self.dfmat = self.data[self.cols].values
        self.padmat = self.padvals[self.cols].values
        self.users = self.data.user_id.unique() 
        self.maxseq = maxseq
        self.has_target = has_target
        self.targetidx =  [self.cols.index(c) for c in \
                           ['answered_correctly', 'user_answer', 'correct_answer', 'content_user_answer']]
        self.padtarget = np.array([self.padvals[self.targetidx].tolist()])
        self.yidx = self.cols.index('answered_correctly') 
        self.timecols = [self.cols.index(c) for c in ['timestamp','prior_question_elapsed_time']]
        self.lagbins = np.concatenate([np.linspace(*a).astype(np.int32) for a in [(0, 10, 6), (12, 100, 45),(120, 600, 80), 
                              (660, 1440, 28), (1960, 10800, 36), (10800, 259200, 60), 
                              (518400, 2592000, 10), (2592000, 31104000, 22), (31104000, 311040000, 10)]])
    
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
        useqidx = useqidx[:cappos][-self.maxseq:]
        umat = self.dfmat[useqidx].astype(np.float32)
        
        useqlen = umat.shape[0]
        if useqlen < self.maxseq:
            padlen = self.maxseq - umat.shape[0] 
            upadmat = np.tile(self.padmat, (padlen, 1))
            umat = np.concatenate((upadmat, umat), 0)
            
        # convert time to lag
        umat[:, self.timecols[0]][1:] = umat[:, self.timecols[0]][1:] - umat[:, self.timecols[0]][:-1]
        umat[:, self.timecols[0]][0] = 0
        # make lag 1 to 5
        lagmat = np.transpose(np.stack( \
                    np.concatenate((umat[:, self.timecols[0]][l:], np.ones(l)*10**8)) \
                        for l in range(1, 5)))
        
        # Time embeddings
        timeemb =   np.stack(( \
                    np.digitize(umat[:, self.timecols[0]]/ 1000, self.lagbins), 
                    (umat[:, self.timecols[1]] / 1000).clip(0, 300))).round()
        timeemb = np.transpose(timeemb, (1,0))
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
        
        # Create mask
        umask = torch.zeros(umat.shape[0], dtype=torch.int8)
        umask[-useqlen:] = 1
        
        return umat, umask, target
    
class LearnNet(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 device = device, dropout = 0.2, model_type = args.model, hidden = args.hidden):
        super(LearnNet, self).__init__()
        
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
        
        IN_UNITS = 32 + 32 + 4 + 16 * (2 + 1) + 5 + len(self.contcols)
        LSTM_UNITS = hidden
        
        if self.model_type == 'lstm':
            self.seqnet = nn.LSTM(IN_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
        if self.model_type == 'gru':
            self.seqnet = nn.GRU(IN_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
        if self.model_type == 'xlm':
            self.xcfg = XLMConfig()
            self.xcfg.causal = True
            self.xcfg.emb_dim = LSTM_UNITS
            self.xcfg.max_position_embeddings = args.maxseq
            self.xcfg.n_layers = args.n_layers
            self.xcfg.n_heads = args.n_heads
            self.xcfg.return_dict = False
            self.seqnet  = XLMModel(self.xcfg)
            
        self.linear1 = nn.Linear(LSTM_UNITS, LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS)
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)
        
    def forward(self, x, m = None):
        
        # Categroical embeddings
        embcat = torch.cat([
            self.emb_content_id(  x[:,:, self.modcols.index('content_id')].long()  ),
            self.emb_bundle_id(  x[:,:, self.modcols.index('bundle_id')].long()  ),
            self.emb_cont_user_answer(  x[:,:, self.modcols.index('content_user_answer')].long()  ),
            self.emb_part(  x[:,:, self.modcols.index('part')].long()  ), 
            (self.emb_tag(x[:,:, self.tag_idx].long()) * self.tag_wts).sum(2),
            self.emb_lag_time(   x[:,:, self.modcols.index('lag_time_cat')].long()   ), 
            self.emb_elapsed_time(  x[:,:, self.modcols.index('elapsed_time_cat')].long()  )
            ] #+ [self.emb_tag(x[:,:, ii.item()].long()) for ii in torch.where(self.tag_idx)[0]]
            , 2)
        embcat = self.embedding_dropout(embcat)
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        # Weighted sum of tags - hopefully good weights are learnt
        xinp = torch.cat([embcat, contmat], 2)
        
        if self.model_type == 'xlm':
            xinp = self.linearx(xinp)
            inputs = {'input_ids': None, 'inputs_embeds': xinp, 'attention_mask': m}
            hidden = self.seqnet(**inputs)
        else:
            hidden, _ = self.seqnet(xinp)
        # Take last hidden unit
        hidden = hidden[:,-1,:]
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out

logger.info('Create model and loaders')
model = self = LearnNet(MODCOLS, CONTCOLS, PADVALS, EXTRACOLS)
model.to(device)
torch.save(model.state_dict(), 'tmp.bin')


# Should we be stepping; all 0's first, then all 1's, then all 2,s 
trndataset = self = SAKTDataset(train, None, MODCOLS, PADVALS, EXTRACOLS, maxseq = args.maxseq)
valdataset = SAKTDataset(valid, train, MODCOLS, PADVALS, EXTRACOLS, maxseq = args.maxseq)
loaderargs = {'num_workers' : args.workers, 'batch_size' : args.batchsize}
trnloader = DataLoader(trndataset, shuffle=True, **loaderargs)
valloader = DataLoader(valdataset, shuffle=False, **loaderargs)
x, m, y = next(iter(trnloader))

'''
from transformers import XLMTokenizer, XLMModel
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
model = XLMModel.from_pretrained("xlm-clm-ende-1024")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# {'input_ids': tensor([[   0, 6496,   15,   52, 2232,   26, 9684,    1]]),
# 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 
#'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
outputs = model(**inputs)
'''

criterion =  nn.BCEWithLogitsLoss()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [ {'params': [p for n, p in param_optimizer] } ]
optimizer = torch.optim.Adam(plist, lr=args.lr)


if device != 'cpu':
    scaler = torch.cuda.amp.GradScaler()


logger.info('Start training')
best_val_loss = 100.
trn_lossls = []
predls = []
bags = 4
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
        x, m, y = batch
        x = x.to(device, dtype=torch.float)
        m = m.to(device, dtype=torch.long)
        y = y.to(device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = torch.autograd.Variable(y)
        '''
        out = model(x, m)
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
        x, m, y = batch
        x = x.to(device, dtype=torch.float)
        m = m.to(device, dtype=torch.long)
        with torch.no_grad():
            out = model(x, m)
        y_predls.append(out.detach().cpu().numpy())
        
    y_pred = np.concatenate(y_predls)
    predls.append(y_pred)
    auc_score = roc_auc_score(y_act, y_pred )
    logger.info(f'Valid AUC Score {auc_score:.5f}')
    auc_score = roc_auc_score(y_act, sum(predls[-bags:]) )
    logger.info(f'Bagged valid AUC Score {auc_score:.5f}')


# Ideas:
# Split tags to separate embeddings
# Try with a gru instead of an lstm
# Part corrent for historical 
# Make the content_user_answer inside the dict (for submission time)
# Store the mean vals inside pdict for submission time. 
# Make an embedding out of the count of user answers and the count of correct