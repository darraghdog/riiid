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
import random
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
from tools.zoo import LearnNet12,LearnNet14, LearnNet20, LearnNet21, LearnNet24
from transformers import XLMModel, XLMConfig
from copy import deepcopy

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
logger = get_logger('Stack', 'INFO')

# funcs for user stats with loop
def add_user_feats(df, pdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
    cu = np.zeros(len(df), dtype=np.uint32)
    #acsb = np.zeros(len(df), dtype=np.uint32)
    #cb = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    contid = np.zeros((len(df), 1), dtype=np.uint8)
    qamat = np.zeros((len(df),2), dtype=np.float32)
    lect = np.zeros((len(df), 2), dtype=np.uint32)
    lectcat = np.zeros((len(df), 2), dtype=np.uint32)

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
            pdicts['count_u_lect_dict'][uidx] += 1
            pdicts['count_u_lect_timestamp'][uidx] = int(round(tstmp / 1000))
            pdicts['lecture_logged'][uidx] = 1
            pdicts['lecture_tag'][uidx] = ldict['tag'][cid] 
            pdicts['lecture_part'][uidx] = ldict['part'][cid]
            continue
        
        try:
            uidx = pdicts['uidx'][u]
        except:
            pdicts['max_uidx'] += 1
            uidx = pdicts['uidx'][u] = pdicts['max_uidx']
            
        try:
            # uqidx = pdicts['uqidx'][(u, cid)]
            uqidx = pdicts['uqidx'][cid][u]
        except:
            pdicts['max_uqidx'] += 1
            # uqidx = pdicts['uqidx'][(u, cid)] = pdicts['max_uqidx']
            uqidx = pdicts['uqidx'][cid][u] = pdicts['max_uqidx']
        
        #bid = bdict[cid]
        #newbid = bid == pdicts['track_b'].item(uidx)
            
        lect[cnt] = pdicts['count_u_lect_dict'].item(uidx), \
            int(round(tstmp / 1000)) - pdicts['count_u_lect_timestamp'].item(uidx)
        lectcat[cnt] = pdicts['lecture_tag'].item(uidx), pdicts['lecture_part'].item(uidx)
        
        acsu[cnt] = pdicts['answered_correctly_sum_u_dict'].item(uidx)
        cu[cnt] = pdicts['count_u_dict'].item(uidx)
        #acsb[cnt] = pdicts['answered_correctly_sum_b_dict'].item(uidx)
        #cb[cnt] = pdicts['count_b_dict'].item(uidx)
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
            #pdicts['count_b_dict'][uidx] = 1 if newbid else pdicts['count_b_dict'][uidx] + 1
            #if newbid : pdicts['answered_correctly_sum_b_dict'][uidx] = 0
            if yprev: 
                pdicts['answered_correctly_sum_u_dict'][uidx] += 1
                #pdicts['answered_correctly_sum_c_dict'][cid] += 1
                #pdicts['answered_correctly_sum_b_dict'][uidx] += 1
                pdicts['content_id_answered_correctly_sum_u_dict'][uqidx] += 1
            #pdicts['track_b'][uidx] = bid
            
            if pdicts['lecture_logged'][uidx] == 1:
                pdicts['lecture_tag'][uidx] = 0
                pdicts['lecture_part'][uidx] = 0
                pdicts['lecture_logged'][uidx] = 1
                
    #countmat = np.transpose(np.stack([cu, cidcu, cb]), (1,0)).astype(np.float32)
    #correctmat = np.transpose(np.stack([acsu,  cidacsu, acsb]), (1,0)).astype(np.float32)
    countmat = np.transpose(np.stack([cu, cidcu]), (1,0)).astype(np.float32)
    correctmat = np.transpose(np.stack([acsu,  cidacsu]), (1,0)).astype(np.float32)
    avgcorrectmat = correctmat / (countmat + 0.001).astype(np.float32)
    acsumat = np.expand_dims(acsu, 1).astype(np.float32)
    lect = lect.astype(np.float32)
    lectcat = lectcat.astype(np.float32)
    outmat = np.concatenate((countmat, avgcorrectmat, acsumat, qamat, lect, lectcat), 1)
    cols = [f'counts___feat{i}' for i in range(2)] + \
                [f'avgcorrect___feat{i}' for i in range(2)] + \
                    ['cid_answered_correctly'] + [f'rank_stats_{i}' for i in range(2)] + \
                        ['lecture_ct','lecture_lag', 'lecture_tag','lecture_part']
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
arg('--maxseq', type=int, default=512)
arg('--hidden', type=int, default=512)
arg('--dumpdata', type=int, default=0)
arg('--loaddata', type=int, default=0)
arg('--bags', type=int, default=4)
arg('--model', type=str, default='lstm')
arg('--infer', type=str, default=0)
arg('--label-smoothing', type=float, default=0.01)
arg('--dir', type=str, default='val')
#arg('--version', type=str, default='V05')
args = parser.parse_args()
args.dumpdata = bool(args.dumpdata)
args.loaddata = bool(args.loaddata)
args.infer = bool(args.infer)
logger.info(args)


device = 'cpu' if platform.system() == 'Darwin' else 'cuda'
CUT=0
DIR=args.dir#'val'
VERSION='V01S'#args.version
debug = False
validaten_flg = False


FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', \
                       'timestamp', 'user_answer']
logger.info(f'Loaded columns {", ".join(FILTCOLS)}')

if args.loaddata:
    logger.info('Load pdicts')
    pdicts = loadobj(f'data/{DIR}/pdicts_{VERSION}_pre.pk')
    logger.info('Load valid')
    valid = loadobj(f'data/{DIR}/valid_{VERSION}_pre.pk')
    logger.info('Load train')
    train = loadobj(f'data/{DIR}/train_{VERSION}_pre.pk')
    logger.info('Done loading....')
else:

    valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]#.head(10**5)
    train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]#.head(2*10**6)
    
    train = train.sort_values(['user_id', 'timestamp']).reset_index(drop = True)
    valid = valid.sort_values(['user_id', 'timestamp']).reset_index(drop = True)
    
    # Joins questions
    ldf = pd.read_csv('data/lectures.csv')
    ldf.type_of = ldf.type_of.str.replace(' ', '_')
    ldict = ldf.set_index('lecture_id').to_dict()
    #lecture_types = [t for t in ldf.type_of.unique() if t!= 'starter']
    
    
    qdf = pd.read_csv('data/questions.csv')
    qdf[[f'tag{i}' for i in range(6)]] =  qdf.tags.fillna('').apply(split_tags).tolist()
    
    bdict = qdf.set_index('question_id')['bundle_id'].to_dict()
    keepcols = ['question_id', 'part', 'bundle_id', 'correct_answer'] + [f'tag{i}' for i in range(6)]
    train = pd.merge(train, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    valid = pd.merge(valid, qdf[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
    formatcols =  ['question_id', 'part', 'bundle_id', 'correct_answer', 'user_answer']+ [f'tag{i}' for i in range(6)]
    train[formatcols] = train[formatcols].fillna(0).astype(np.int16)
    valid[formatcols] = valid[formatcols].fillna(0).astype(np.int16)
    
    
    '''
    train.loc[train.content_type_id == False]['question_id'].value_counts().head(1000).sum()
    train.loc[train.content_id == False]['question_id']
    ix = train.content_type_id == False
    a  =pd.crosstab(train[ix].user_id, train[ix].answered_correctly).sort_values(1)
    a['count'] = a.sum(1)
    a['avg'] = a[1].values / ( a[1].values  +  a[0].values )
    a.sort_values('count', ascending = False).head(500)
    a[(a['count']>1000)].avg.mean()
    a[(a['count']>1000)]['count'].sum()
    '''
    
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
    
    
    # Count task container id
    taskcols = ['user_id', 'task_container_id']
    train['task_container_cts'] = train[taskcols][::-1].groupby(taskcols).cumcount()[::-1]
    valid['task_container_cts'] = valid[taskcols][::-1].groupby(taskcols).cumcount()[::-1]
    
    
    # user stats features with loops
    qidx = train.content_type_id == False
    n_users = int(len(train[qidx].user_id.unique()) * 1.2)
    n_users_ques = int(len(train[qidx][['user_id', 'content_id']].drop_duplicates()) * 1.2)
    u_int_cols = ['answered_correctly_sum_u_dict', 'count_u_dict', 'lecture_tag', 'lecture_part', 'lecture_logged', \
                  'content_id_lag',  'pexp_count_u_dict', 'count_u_lect_dict', 'count_u_lect_timestamp'] #'track_b', 'answered_correctly_sum_b_dict',  'count_b_dict', 
    u_float_cols = ['userRatioCum', 'userAvgRatioCum', 'qaRatiocum', 'qaRatioCorrectcum', ]
    uq_int_cols = ['content_id_answered_correctly_sum_u_dict', 'content_id_count_u_dict']
    
    
    pdicts =  {**dict((col, np.zeros(n_users, dtype= np.uint32)) for col in u_int_cols), 
             **dict((col, np.zeros(n_users, dtype= np.float32)) for col in u_float_cols), 
             **dict((col, np.zeros(n_users_ques, dtype= np.uint8)) for col in uq_int_cols), 
             **{'qaRatio' : qaRatio, 'qaCorrect': qaCorrect}}
    
    
    cid_udict = train[qidx][['user_id', 'content_id']].drop_duplicates() \
                                .reset_index(drop=True).reset_index().groupby('content_id') \
                                .apply(lambda x : x.set_index('user_id')['index'].to_dict()  )
    pdicts['uqidx'] = 13523 * [{}]
    for id_,row_ in cid_udict.iteritems():
        pdicts['uqidx'][id_] = row_
    del cid_udict
    pdicts['max_uqidx'] = max(max(d.values()) for d in pdicts['uqidx'] if d!= {})
    # pdicts['uqidx'] = train[qidx][['user_id', 'content_id']].drop_duplicates() \
    #            .reset_index(drop = True).reset_index() \
    #                .set_index(['user_id', 'content_id']).to_dict()['index']
    pdicts['uidx'] = train[qidx][['user_id']].drop_duplicates() \
                .reset_index(drop = True).reset_index() \
                    .set_index(['user_id']).to_dict()['index']
    pdicts['max_uidx'] = max(v for v in pdicts['uidx'].values())


    train = add_user_feats(train, pdicts)
    valid = add_user_feats(valid, pdicts)

    # For start off remove lectures
    train = train.loc[train.content_type_id == False].reset_index(drop=True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)
    
    train['content_user_answer']  = train['user_answer'] + 4 * train['content_id'].astype(np.int32)
    valid['content_user_answer']  = valid['user_answer'] + 4 * valid['content_id'].astype(np.int32)
    #train['answered_correctly_ct_log'] = np.log1p(train['answered_correctly_ct_c'].fillna(0).astype(np.float32)) - 7.5
    #valid['answered_correctly_ct_log'] = np.log1p(valid['answered_correctly_ct_c'].fillna(0).astype(np.float32)) - 7.5
    
    pdicts['NORMCOLS'] = ['counts___feat0', 'counts___feat1', 'cid_answered_correctly', 
                          'lecture_ct','lecture_lag', 'answered_correctly_ct_c']
    meanvals = np.log1p(train[pdicts['NORMCOLS']].fillna(0).astype(np.float32)).mean().values
    stdvals = np.log1p(train[pdicts['NORMCOLS']].fillna(0).astype(np.float32)).std().values
    
    pdicts['meanvals'] = meanvals
    pdicts['stdvals'] = stdvals
    train[pdicts['NORMCOLS']] = (np.log1p(train[pdicts['NORMCOLS']].fillna(0).astype(np.float32)) - meanvals) / stdvals
    valid[pdicts['NORMCOLS']] = (np.log1p(valid[pdicts['NORMCOLS']].fillna(0).astype(np.float32)) - meanvals) / stdvals
    
    # Create index for loader
    trnidx = train.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()
    validx = valid.reset_index().groupby(['user_id'])['index'].apply(list).to_dict()
    
    FEATCOLS = ['counts___feat0', 'avgcorrect___feat0', 'counts___feat1', 'avgcorrect___feat1', 
        'cid_answered_correctly', 'rank_stats_0', 'rank_stats_1'] #
    
    
    pdicts['MODCOLS'] = ['content_id', 'content_type_id', 'prior_question_elapsed_time', \
               'prior_question_had_explanation', 'task_container_id', 'lecture_tag', 'lecture_part', \
                'timestamp', 'part', 'bundle_id', 'task_container_cts', \
                    'answered_correctly', 'user_answer', 'correct_answer', 'content_user_answer', 
                    'answered_correctly_avg_c', 'answered_correctly_ct_c', 'attempts_avg_c', 'lecture_ct','lecture_lag'] \
                + [f'tag{i}' for i in range(6)] + FEATCOLS
    
    # EMBCOLS = ['content_id', 'part', 'bundle_id'] + [f'tag{i}' for i in range(6)]
    pdicts['TARGETCOLS'] = [ 'user_answer', 'answered_correctly', 'correct_answer', 'content_user_answer']
    
    # SHIFT TARGET HERE
    pdicts['CARRYTASKFWD'] = ['counts___feat0', 'avgcorrect___feat0',  \
                    'cid_answered_correctly', 'rank_stats_0', 'rank_stats_1'] #
    
    pdicts['CONTCOLS'] = ['timestamp', 'prior_question_elapsed_time', 'prior_question_had_explanation', \
                'answered_correctly_avg_c', 'answered_correctly_ct_c', 'attempts_avg_c', \
                'task_container_cts', 'lecture_ct','lecture_lag'] + FEATCOLS
    pdicts['NOPAD'] = ['prior_question_elapsed_time', 'prior_question_had_explanation', \
                 'timestamp', 'content_type_id', 'task_container_cts'] + pdicts['CONTCOLS']
        
    pdicts['PADVALS'] = train[pdicts['MODCOLS']].max(0) + 1
    pdicts['PADVALS'][pdicts['NOPAD']] = 0
    pdicts['EXTRACOLS'] = ['lag_time_cat',  'elapsed_time_cat']
    
    #self = SAKTDataset(train, MODCOLS, PADVALS)
    pdicts['keepcols'] = keepcols
    pdicts['content_df'] = content_df
    pdicts['ldict'] = ldict
    pdicts['bdict'] = bdict
    pdicts['qdf'] = qdf

if args.dumpdata:
    dumpobj(f'data/{DIR}/pdicts_{VERSION}_pre.pk', pdicts)
    dumpobj(f'data/{DIR}/valid_{VERSION}_pre.pk', valid)
    dumpobj(f'data/{DIR}/train_{VERSION}_pre.pk', train)
    
    
class SAKTDataset(Dataset):
    def __init__(self, data, basedf, cols, padvals, extracols, carryfwdcols, 
                 maxseq = args.maxseq, has_target = True, submit = False): 
        super(SAKTDataset, self).__init__()
        
        self.cols = cols
        self.extracols = extracols
        self.carryfwd = carryfwdcols
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
        self.quidxbackup = self.quidx.copy()
        
        #if basedf is None:
        #    self.quidx = self.quidx[np.random.choice(self.quidx.shape[0], 2*10**6, replace=False)]
        
        self.task_container_id = self.data.task_container_id.values
        self.carryfwdidx = [self.cols.index(c) for c in self.carryfwd]
        
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
    
    def __getitem__(self, idx, row = None):
        
        if self.submit:
            # in this case, user will be the index, otherwise, we will pass the id
            u = idx
            umatls = []
            if u in self.uidx:
                useqidx = self.uidx[u]
                useqidx = useqidx[-self.maxseq:]
                umatls.append(self.dfmat[useqidx].astype(np.float32))
            if u in self.test_matu:
                umatls.append(self.test_matu[u])
            if len(umatls) > 0:
                umatls.append(np.expand_dims(row, 0))
                umat = np.concatenate(umatls)
            else:
                umat = np.expand_dims(row, 0)
            umat = umat[-self.maxseq:]
            
        else: 
            # Get index of user and question
            u,q = self.quidx[idx]
            # Pull out ths user index sequence
            useqidx = self.uidx[u]
            # Pull out position of question
            cappos  = useqidx.index(q) + 1
            # Pull out the sequence of questions up to that question
            container_buffer = 6
            useqidx = useqidx[:cappos][-self.maxseq-container_buffer:]
            # Randomise task container id sequence, but keep the sequence of the last 4
            tstmp = self.dfmat[useqidx,self.cols.index('timestamp')].copy()
            useqidx =  np.array(useqidx) 
            keepstatic = 4
            if len(useqidx)>keepstatic:
                useqidx[:-keepstatic] = useqidx[:-keepstatic][tstmp[:-keepstatic].argsort()]
            # Pull out the sequence for the user
            umat = self.dfmat[useqidx].astype(np.float32)
            
            # Add dummy for task container
            dummy_answer = self.task_container_id[useqidx[-1]] == self.task_container_id[useqidx]
            if sum(dummy_answer) > 1:
                # If we are past the first row of a container; remove 
                #   the previous questions and carry the carryfwd leaky cols
                ffwdvals = umat[dummy_answer][0, self.carryfwdidx]
                # Drop the prevous questions in the container
                dummy_answer[-1] = False
                umat = umat[~dummy_answer]
                umat[-1, self.carryfwdidx] = ffwdvals
            # Now limit to maxseq
            umat = umat[-self.maxseq:]
        
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

# Should we be stepping; all 0's first, then all 1's, then all 2,s 
logger.info('Create loader')
pdicts['daargs'] = daargs = {'cols':pdicts['MODCOLS'], 
          'padvals':pdicts['PADVALS'], 
          'carryfwdcols': pdicts['CARRYTASKFWD'],
          'extracols':pdicts['EXTRACOLS'], 
          'maxseq': args.maxseq}
valdataset = SAKTDataset(valid, train, **daargs)
loaderargs = {'num_workers' : args.workers, 'batch_size' : args.batchsize}
valloader = DataLoader(valdataset, shuffle=False, **loaderargs)
x, m, y = next(iter(valloader))


laargs = {'modcols':pdicts['MODCOLS'], 
          'contcols':pdicts['CONTCOLS'], 
          'padvals':pdicts['PADVALS'], 
          'extracols':pdicts['EXTRACOLS'], 
          'device': device, 
          'dropout': 0.2, 
          'model_type' : 'lstm', 
          'hidden' : 512}
# model = self = LearnNet(**maargs)
# model.to(device)
logger.info('Load weights')
def load_model_weights(modfn, wtname, laargs):
    logger.info(f'load model version {modfn}, weights {wtname}')
    model = modfn(**laargs)
    model.to(device)
    checkpoint = torch.load(wtname,  map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model = model.eval()
    return model

modfns = [LearnNet12, LearnNet20, LearnNet21, LearnNet24]
wtnames = [f'data/{DIR}/{VERSION}/basemodels/lstm_V12_hidden512_ep4.bin', 
           f'data/{DIR}/{VERSION}/basemodels/lstm_V20_hidden512_ep4.bin', 
           f'data/{DIR}/{VERSION}/basemodels/lstm_V20_hidden512_ep7.bin', 
           f'data/{DIR}/{VERSION}/basemodels/lstm_V21_hidden512_ep3.bin', 
           f'data/{DIR}/{VERSION}/basemodels/lstm_V24_hidden512_ep3.bin']
mkeys = ['V12', 'V20A', 'V20B', 'V21', 'V24']
modeldict = dict((k,load_model_weights(modfn, wtname, laargs)) \
                 for (k,modfn, wtname) in zip(mkeys,modfns, wtnames))
wts14 = f'data/{DIR}/{VERSION}/basemodels/lstm_valfull_V14_hidden512_ep12.bin'
laargs14 = deepcopy(laargs)
laargs14['maxseq'] = 128
modeldict['V14'] = load_model_weights(LearnNet14, wts14, laargs14)
# Sort to keep order constent
modeldict = OrderedDict((k,modeldict[k]) for k in sorted(modeldict.keys()))

logger.info('Start inference')
best_val_loss = 100.
predls = []
pbarval = tqdm(enumerate(valloader), 
            total = len(valdataset)//loaderargs['batch_size'], 
            desc=f"Valid ", ncols=0)
y_predls = []
y_act = valid['answered_correctly'].values
contidx = modeldict['V12'].cont_idx
contcols = modeldict['V12'].contcols

for step, batch in pbarval:
    x, m, y = batch
    preddfb = pd.DataFrame(x[:, -1, contidx].detach().cpu().numpy(), columns = contcols)
    x = x.to(device, dtype=torch.float)
    m = m.to(device, dtype=torch.long)
    for k, model in modeldict.items():
        if k=='V14':
            with torch.no_grad(): out = model(x[:,-128:], m[:,-128:], device )
        else:
            with torch.no_grad(): out = model(x, m)
        preddfb[f'pred{k}'] = out.detach().cpu().numpy()
    y_predls.append(preddfb)
    
preddf = pd.concat(y_predls, 0)
preddf['yact'] = y_act

logger.info(f'Preddf head \n {preddf.head()}')
logger.info(f"Preddf correlations \n {preddf.filter(like='pred').corr()}")

for col, colpred in preddf.filter(like='pred').iteritems():
    auc_score = roc_auc_score(y_act, colpred )
    logger.info(f'Valid column {col} AUC Score {auc_score:.5f}')
outfile = f'data/{DIR}/preddf_lvl1_{VERSION}.pk'
dumpobj(outfile, preddf)
logger.info(f'Preds dumped to {outfile}')



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim = 128, dropout = 0.2):
        super(MLP, self).__init__()
                
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.cont_wts = nn.Parameter( torch.ones(self.in_dim))
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        self.linear_out = nn.Linear(self.hidden_dim//2, 1)
        self.bn = nn.BatchNorm1d(num_features=self.in_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        hidden = self.bn(x)
        hidden = hidden * self.cont_wts
        
        hidden = self.dropout( hidden )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout( hidden )
        hidden  = F.relu(self.linear2(hidden))
        out = self.linear_out(hidden).flatten()

        return out

preddf =  loadobj(f'data/{DIR}/preddf_lvl1_{VERSION}.pk')
yact = torch.tensor(preddf.yact.values).float()
preddf = preddf.drop('yact', 1)
alldf = torch.tensor(preddf.values).float()
# alldf = (alldf - alldf.mean(0))/(alldf.std(0))
cut = int(1.5*10**6)
Xtrn = alldf[:cut]
Xval = alldf[cut:]
ytrn = yact[:cut]
yval = yact[cut:]

model = self = MLP(in_dim = Xtrn.shape[1])

criterion =  nn.BCEWithLogitsLoss()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
plist = [ {'params': [p for n, p in param_optimizer] } ]
optimizer = torch.optim.Adam(plist, lr=args.lr)

for col, colpred in preddf.filter(like='pred').iteritems():
    auc_score = roc_auc_score(yact[cut:], colpred [cut:])
    logger.info(f'Valid column {col} AUC Score {auc_score:.5f}')

for epoch in range(args.epochs):
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    shuffled_list = random.sample(range(Xtrn.shape[0]), k=Xtrn.shape[0])
    pbartrn = tqdm(enumerate(chunks(shuffled_list, args.batchsize)), 
                    total = len(shuffled_list)//args.batchsize, 
                    desc=f"Train epoch {epoch}", ncols=0)
    trn_loss = 0.
    for step, chunk in pbartrn:
        optimizer.zero_grad()
        x = Xtrn[chunk].to(device, dtype=torch.float)
        y = ytrn[chunk].to(device, dtype=torch.float)
        x = torch.autograd.Variable(x, requires_grad=True)
        y = torch.autograd.Variable(y)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        pbartrn.set_postfix({'train loss': trn_loss / (step + 1)})
    model.eval()
    with torch.no_grad():
        ypred = model(Xval.to(device, dtype=torch.float))
    auc_score = roc_auc_score(yval.numpy(), ypred.detach().cpu().numpy()   )
    logger.info(f'Valid AUC Score {auc_score:.5f}')



