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
pd.set_option('display.float_format', lambda x: '%.3f' % x)
logger = get_logger('Train', 'INFO')

# funcs for user stats with loop
def add_user_feats(df, pdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
    cu = np.zeros(len(df), dtype=np.uint32)
    acsb = np.zeros(len(df), dtype=np.uint32)
    cb = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    contid = np.zeros((len(df), 1), dtype=np.uint8)
    qamat = np.zeros((len(df),2), dtype=np.float32)

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
        
        try:
            uidx = pdicts['uidx'][u]
        except:
            pdicts['max_uidx'] += 1
            uidx = pdicts['uidx'][u] = pdicts['max_uidx']
            
        try:
            uqidx = pdicts['uqidx'][(u, cid)]
        except:
            pdicts['max_uqidx'] += 1
            uqidx = pdicts['uqidx'][(u, cid)] = pdicts['max_uqidx']
        
        bid = bdict[cid]
        newbid = bid == pdicts['track_b'].item(uidx)
        
        ucid = f'{u}__{cid}'
        acsu[cnt] = pdicts['answered_correctly_sum_u_dict'].item(uidx)
        cu[cnt] = pdicts['count_u_dict'].item(uidx)
        acsb[cnt] = pdicts['answered_correctly_sum_b_dict'].item(uidx)
        cb[cnt] = pdicts['count_b_dict'].item(uidx)
        cidacsu[cnt] = pdicts['content_id_answered_correctly_sum_u_dict'].item(uqidx)
        cidcu[cnt] = pdicts['content_id_count_u_dict'].item(uqidx)
        qamat[cnt] = pdicts['qaRatiocum'].item(uidx) / (pdicts['count_u_dict'].item(uidx) + 0.01), \
                pdicts['qaRatiocum'].item(uidx) / (pdicts['qaRatioCorrectcum'].item(uidx) + 0.01)

        if update:
            pdicts['count_u_dict'][uidx] += 1
            try:
                pdicts['qaRatioCorrectcum'][uidx] += pdicts['qaRatio'][(cid, pdicts['qaCorrect'][cid])]
            except:
                pdicts['qaRatioCorrectcum'][uidx] += 1.
            try:
                pdicts['qaRatiocum'][uidx] += pdicts['qaRatio'][(cid, ua)]
            except:
                pdicts['qaRatiocum'][uidx] += 0.1
                    
            #pdicts['count_c_dict'][cid] += 1
            pdicts['content_id_count_u_dict'][uqidx] += 1
            pdicts['count_b_dict'][uidx] = 1 if newbid else pdicts['count_b_dict'][uidx] + 1
            if newbid : pdicts['answered_correctly_sum_b_dict'][uidx] = 0
            if yprev: 
                pdicts['answered_correctly_sum_u_dict'][uidx] += 1
                #pdicts['answered_correctly_sum_c_dict'][cid] += 1
                pdicts['answered_correctly_sum_b_dict'][uidx] += 1
                pdicts['content_id_answered_correctly_sum_u_dict'][uqidx] += 1
            pdicts['track_b'][uidx] = bid
                
    countmat = np.transpose(np.stack([cu, cidcu, cb]), (1,0)).astype(np.float32)
    correctmat = np.transpose(np.stack([acsu,  cidacsu, acsb]), (1,0)).astype(np.float32)
    avgcorrectmat = correctmat / (countmat + 0.001).astype(np.float32)
    acsumat = np.expand_dims(acsu, 1).astype(np.float32)
    outmat = np.concatenate((countmat, avgcorrectmat, acsumat, qamat), 1)
    cols = [f'counts___feat{i}' for i in range(3)] + \
                [f'avgcorrect___feat{i}' for i in range(3)] + \
                    ['cid_answered_correctly'] + [f'rank_stats_{i}' for i in range(2)]
    outdf = pd.DataFrame(outmat, columns = cols, index = df.index.tolist())
    df = pd.concat([df, outdf], 1)
    
    return df

DECAY = 0.0
logger.info('Load args')
parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
arg = parser.add_argument
arg('--workers', type=int, default=8, help='number of cpu threads to use')
arg('--batchsize', type=int, default=1024)
arg('--lr', type=float, default=0.001)
arg('--epochs', type=int, default=12)
arg('--maxseq', type=int, default=128)
arg('--hidden', type=int, default=256)
arg('--n_layers', type=int, default=2)
arg('--n_heads', type=int, default=8)
arg('--dumpdata', type=bool, default=0)
arg('--bags', type=int, default=4)
arg('--model', type=str, default='lstm')
arg('--label-smoothing', type=float, default=0.01)
arg('--dir', type=str, default='val')
#arg('--version', type=str, default='V05')
args = parser.parse_args()
args.dumpdata = bool(args.dumpdata)
logger.info(args)


device = 'cpu' if platform.system() == 'Darwin' else 'cuda'
CUT=0
DIR=args.dir#'val'
VERSION='V07'#args.version
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
    aggdf['answerratio'] = (aggdf.answcount / aggdf.quescount).astype(np.float32)
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
                .groupby(['content_id']).agg(['mean', 'count']).astype(np.float32).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c', 'answered_correctly_ct_c']
content_df2 = train.query('content_type_id == 0') \
                .groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float32).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df.columns
del content_df1, content_df2
gc.collect()

content_df.iloc[:,1:] = content_df.iloc[:,1:].astype(np.float32)
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

'''
# user stats features with loops
pdicts = {'answered_correctly_sum_u_dict' : defaultdict(int),
          'count_u_dict' : defaultdict(int),
          'answered_correctly_sum_b_dict' : defaultdict(int),
          'count_b_dict' : defaultdict(int),
          'track_b' : defaultdict(int),
          'content_id_answered_correctly_sum_u_dict' : defaultdict(int),
          'content_id_count_u_dict': defaultdict(int),
          'userAvgRatioCum': defaultdict(float),
          'userRatioCum' : defaultdict(int),
          
          'qaRatio' : qaRatio,
          'qaCorrect': qaCorrect,
          'qaRatiocum' : defaultdict(int),
          'qaRatioCorrectcum' : defaultdict(int)}
'''

# user stats features with loops
n_users = int(len(train.user_id.unique()) * 1.2)
n_users_ques = int(len(train[['user_id', 'content_id']].drop_duplicates()) * 1.2)
u_int_cols = ['answered_correctly_sum_u_dict', 'count_u_dict', 'answered_correctly_sum_b_dict', \
               'count_b_dict', 'track_b', 'content_id_lag',  'pexp_count_u_dict']
u_float_cols = ['userRatioCum', 'userAvgRatioCum', 'qaRatiocum', 'qaRatioCorrectcum', ]
uq_int_cols = ['content_id_answered_correctly_sum_u_dict', 'content_id_count_u_dict']


pdicts =  {**dict((col, np.zeros(n_users, dtype= np.uint32)) for col in u_int_cols), 
         **dict((col, np.zeros(n_users, dtype= np.float32)) for col in u_float_cols), 
         **dict((col, np.zeros(n_users_ques, dtype= np.uint32)) for col in uq_int_cols), 
         **{'qaRatio' : qaRatio, 'qaCorrect': qaCorrect}}

pdicts['uqidx'] = train[['user_id', 'content_id']].drop_duplicates() \
            .reset_index(drop = True).reset_index() \
                .set_index(['user_id', 'content_id']).to_dict()['index']
pdicts['uidx'] = train[['user_id']].drop_duplicates() \
            .reset_index(drop = True).reset_index() \
                .set_index(['user_id']).to_dict()['index']
pdicts['max_uidx'] = max(v for v in pdicts['uidx'].values())
pdicts['max_uqidx'] = max(v for v in pdicts['uqidx'].values())


train = add_user_feats(train, pdicts)
if args.dumpdata:
    dumpobj(f'data/{DIR}/pdicts_{VERSION}_pre.pk', pdicts)
    dumpobj(f'data/{DIR}/valid_{VERSION}_pre.pk', valid)
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
pdicts['meanvals'] = meanvals
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

'''
# Save objects
FILTCOLS, FEATCOLS, EMBCOLS, TARGETCOLS, NOPAD, 
ldict
bdict
qdf
'''
pdicts['MODCOLS'], pdicts['PADVALS'], pdicts['EXTRACOLS'], pdicts['CONTCOLS'] = MODCOLS, PADVALS, EXTRACOLS, CONTCOLS
pdicts['content_df'] = content_df
pdicts['ldict'] = ldict
pdicts['bdict'] = bdict
pdicts['qdf'] = qdf

if args.dumpdata:
    logger.info('Dump objects - pdicts')
    for k, v in pdicts.items():
        dumpobj(f'data/{DIR}/pdicts____{VERSION}_{k}.pk', v)
    uqidxll = [(*k,v) for k,v in tqdm(pdicts['uqidx'].items())]
    fo = open(f'data/{DIR}/pdicts____{VERSION}_uqidx.csv','w')
    for l in tqdm(uqidxll, total = len(uqidxll)):
        s = '%s %s %s\n'%(l)
        fo.write(s)
    fo.close()	
    '''
    fo = open(f'data/{DIR}/{VERSION}/pdicts____uqidx.csv','r')
    uqidx = defaultdict(lambda: {})
    for t, l in tqdm(enumerate(fo)):
        l = list(map(int, l[:-1].split()))
        uqidx[l[0]][l[1]] = l[2]
        l = None
    fo.close()
    '''
    logger.info('Dump objects - train/val')
    dumpobj(f'data/{DIR}/train_{VERSION}.pk', train)
    dumpobj(f'data/{DIR}/valid_{VERSION}.pk', valid)
    logger.info('Done... yayy!!')
    gc.collect()

logger.info(f'Na vals train \n\n{train.isna().sum()}')
logger.info(f'Na vals valid \n\n{valid.isna().sum()}')
logger.info(f'Max vals train \n\n{train.max()}')
logger.info(f'Max vals valid \n\n{valid.max()}')

class SAKTDataset(Dataset):
    def __init__(self, data, basedf, cols, padvals, extracols, 
                 maxseq = args.maxseq, has_target = True, submit = False): 
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
        
        self.data[['timestamp','prior_question_elapsed_time']] = \
            self.data[['timestamp','prior_question_elapsed_time']] / 1000
        self.dfmat = self.data[self.cols].values.astype(np.float32)
        self.users = self.data.user_id.unique() 
        del self.data
        gc.collect()
        
        self.padmat = self.padvals[self.cols].values
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
        self.submit = False
    
    def __len__(self):
        
        return len(self.quidx)
    
    def __getitem__(self, idx):
        
        if self.submit:
            # in this case, user will be the index, otherwise, we will pass the id
            useqidx = self.uidx[u]
            # Pull out the max sequence of questions 
            useqidx = useqidx[-self.maxseq:]
            umat = self.dfmat[useqidx].astype(np.float32)
            
        else: 
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
        
        # Time embeddings
        timeemb =   np.stack(( \
                    np.digitize(umat[:, self.timecols[0]], self.lagbins), 
                    (umat[:, self.timecols[1]]).clip(0, 300))).round()
        timeemb = np.transpose(timeemb, (1,0))
        umat = np.concatenate((umat, timeemb), 1)
        
        # preprocess continuous time - try log scale and roughly center it
        umat[:, self.timecols] = np.log10( 1.+ umat[:, self.timecols] / 60  ) 
        
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
                 device = device, dropout = 0.2, model_type = args.model, 
                 hidden = args.hidden):
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
        
        IN_UNITS = \
                self.emb_content_id.embedding_dim + self.emb_bundle_id.embedding_dim + \
                self.emb_part.embedding_dim + self.emb_tag.embedding_dim + \
                self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim + \
                len(self.contcols)
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
model = LearnNet(pdicts['MODCOLS'], pdicts['CONTCOLS'], pdicts['PADVALS'], pdicts['EXTRACOLS'])
model.to(device)

# Should we be stepping; all 0's first, then all 1's, then all 2,s 
trndataset = SAKTDataset(train, None, pdicts['MODCOLS'], pdicts['PADVALS'], pdicts['EXTRACOLS'], maxseq = args.maxseq)
valdataset = SAKTDataset(valid, train, pdicts['MODCOLS'], pdicts['PADVALS'], pdicts['EXTRACOLS'], maxseq = args.maxseq)
loaderargs = {'num_workers' : args.workers, 'batch_size' : args.batchsize}
trnloader = DataLoader(trndataset, shuffle=True, **loaderargs)
valloader = DataLoader(valdataset, shuffle=False, **loaderargs)
x, m, y = next(iter(trnloader))

# Prep class for inference
if args.dumpdata:
    logger.info('Dump objects - tail of maxseq')
    df = pd.concat([train, valid]).reset_index(drop = True)
    df = df.sort_values(['user_id', 'timestamp']).groupby(['user_id']).tail(args.maxseq)
    dumpobj(f'data/{DIR}/train_all_{VERSION}_tail.pk', df)
    alldataset = SAKTDataset(trainall, None, pdicts['MODCOLS'], pdicts['PADVALS'], pdicts['EXTRACOLS'], maxseq = args.maxseq)
    dumpobj(f'data/{DIR}/alldataset_{VERSION}_tail.pk', df)
    del df, alldataset 
    gc.collect()

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
bags = args.bags
for epoch in range(args.epochs):
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
        
        out = model(x, m)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        '''
        with autocast():
            out = model(x, m)
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
    torch.save(model.state_dict(), f'data/{DIR}/{args.model}_{VERSION}_hidden{args.hidden}_ep{epoch}.bin')
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