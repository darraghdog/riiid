# https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering/#data
import os
os.chdir('/Users/dhanley/Documents/riiid/')
import sys
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import lightgbm as lgb
import warnings
from scipy import sparse
from scripts.utils import Iter_Valid, dumpobj, loadobj
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',1000)

pd.set_option('display.width', 1000)

def hrfn(val):
    return int(round(val/(3600*1000) % 24))
def dayfn(val):
    return int(round(val/(3600*1000*24) % 7))

# funcs for user stats with loop
def add_user_feats(df, pdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
    acsudec = np.zeros((len(df),2), dtype=np.float16)
    cu = np.zeros(len(df), dtype=np.uint32)
    acsb = np.zeros(len(df), dtype=np.uint32)
    cb = np.zeros(len(df), dtype=np.uint32)
    expacsu = np.zeros(len(df), dtype=np.uint32)
    expcu = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    tstamp = np.zeros((len(df), 5), dtype=np.uint32)
    tstavg = np.zeros(len(df), dtype=np.float32)
    #ctunq = np.zeros((len(df), 2), dtype=np.uint32)
    lectct = np.zeros((len(df), 7), dtype=np.uint32)
    lectavg = np.zeros((len(df), 3), dtype=np.float32)
    pexpm = np.zeros((len(df), 1), dtype=np.uint8)
    contid = np.zeros((len(df), 1), dtype=np.uint8)
    qamat = np.zeros((len(df),6), dtype=np.float16)
    nattmpt = np.zeros((len(df),4), dtype=np.float16)
    bratio = np.zeros((len(df),3), dtype=np.float16)

    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})

    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp', 'content_type_id', 'user_answer']
    if not update:
        itercols = [f for f in itercols if f!='answered_correctly']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp, ctype, ua = row
        else:
            u, pexp, eltim, cid, tag, tcid, tstmp, ctype, ua = row
        
        if ctype==1:
            lectcid, lpart, ltype_of = cid, ldict['part'][cid], ldict['type_of'][cid]
            pdicts['cum_lecture_ct'][u] += 1
            pdicts['prev_lecture_part'][u] = lpart
            pdicts['prev_lecture_timestamp3'][u] = pdicts['prev_lecture_timestamp2'][u]
            pdicts['prev_lecture_timestamp2'][u] = pdicts['prev_lecture_timestamp'][u]
            pdicts['prev_lecture_timestamp'][u] = tstmp 
            pdicts['cum_ques_since_lecture'][u] = 1
            continue
        else:
            
            if pdicts['cum_ques_since_lecture'][u] == 1:
                time_since = int(round(tstmp - pdicts['prev_lecture_timestamp'][u]))/1000
                pdicts['cum_lecture_time'][u] += time_since
                pdicts['prev_lecture_time'][u] = time_since
            if pdicts['cum_ques_since_lecture'][u]>0:
                pdicts['cum_ques_since_lecture'][u] += 1
        '''
        if cid + 1 == pdicts['ques_curr'][u]:
            pdicts['ques_reattempt_count'][u] += 1
        else:
            pdicts['ques_reattempt_count_lag1'][u] = pdicts['ques_reattempt_count'][u]
            #pdicts['ques_lag1'][u] = pdicts['ques_curr'][u]
            pdicts['ques_curr'][u] = cid + 1
            pdicts['ques_reattempt_count'][u] = 0
        '''
        
        if (tcid + 1 == pdicts['container_curr'][u]) or (cid in pdicts['container_unq_ques'][u]):
            if cid in pdicts['container_unq_ques'][u]:
                pdicts['container_curr'][u] = tcid + 1
            pdicts['container_reattempt_count'][u] += 1
            pdicts['container_unq_ques'][u].add(cid)
        else:
            pdicts['container_reattempt_count_lag1'][u] = pdicts['container_reattempt_count'][u]
            #pdicts['ques_lag1'][u] = pdicts['ques_curr'][u]
            pdicts['container_unq_ques_lag1'][u] = len(pdicts['container_unq_ques'][u])
            pdicts['container_curr'][u] = tcid + 1
            pdicts['container_reattempt_count'][u] = 1
            pdicts['container_unq_ques'][u] = set([cid])
            pdicts['container_lag2'][u] = pdicts['container_lag1'][u]
            pdicts['container_lag1'][u] = pdicts['container_lag0'][u]
            pdicts['container_lag0'][u] = (tstmp /1000)
        
        bid = bdict[cid]
        newbid = bid == pdicts['track_b'][u]
        
        if pdicts['count_u_dict'][u] == 0:
            pdicts['bundcurr'][u] = bid
        else:
            if bid!=pdicts['bundcurr'][u]:
                pdicts['bundprev'][u] = pdicts['bundcurr'][u]
                pdicts['bundcurr'][u] = bid
            if pdicts['count_u_dict'][u] > 1:
                pdicts['bundpathcum'][u] += pdicts['bundprevtime'][pdicts['bundprev'][u]]
                pdicts['bundusercum'][u] += eltim
        
        bratio[cnt] = pdicts['bundpathcum'][u] / (pdicts['bundusercum'][u]  + 0.01), \
                            eltim / (pdicts['bundprevtime'][pdicts['bundprev'][u]] + 0.01), \
                            pdicts['userRatioCum'][u] / (pdicts['userAvgRatioCum'][u] + 0.01)
        
        nattmpt[cnt] = pdicts['container_reattempt_count'][u] / len(pdicts['container_unq_ques'][u]), \
                            pdicts['container_reattempt_count_lag1'][u] / (pdicts['container_unq_ques_lag1'][u] + 0.01), \
                            (tstmp /1000) - pdicts['container_lag2'][u], \
                            (tstmp /1000) - pdicts['container_lag1'][u]
                                
                                
        # 4 -> 6 -> 3 -> 2 -> 0 -> 1
        lectct[cnt] = pdicts['cum_lecture_ct'][u], pdicts['prev_lecture_part'][u], \
                    pdicts['prev_lecture_time'][u], pdicts['cum_ques_since_lecture'][u], \
                    int(round(tstmp - pdicts['prev_lecture_timestamp'][u]))/1000, \
                    int(round(tstmp - pdicts['prev_lecture_timestamp2'][u]))/1000, \
                    int(round(tstmp - pdicts['prev_lecture_timestamp3'][u]))/1000
        lectavg[cnt] = pdicts['cum_lecture_time'][u] / (pdicts['cum_lecture_ct'][u] + 0.01), \
                        pdicts['cum_lecture_ct'][u] / ( pdicts['count_u_dict'][u] +0.01), \
                        pdicts['cum_ques_since_lecture'][u] / ( pdicts['count_u_dict'][u] +0.01)
                                 
        ucid = f'{u}__{cid}'
        acsu[cnt] = pdicts['answered_correctly_sum_u_dict'][u]
        if pdicts['count_u_decay_dict'][u] > 0:
            decay_ratio = pdicts['answered_correctly_sum_u_decay_dict'][u] / pdicts['count_u_decay_dict'][u]
            decay_ratio1 = pdicts['answered_correctly_sum_u_decay_dict1'][u] / pdicts['count_u_decay_dict1'][u]
            acsudec[cnt] = decay_ratio, \
                            decay_ratio - (pdicts['answered_correctly_sum_u_dict'][u]/  pdicts['count_u_dict'][u])
                    
                        
        cu[cnt] = pdicts['count_u_dict'][u]
        acsb[cnt] = pdicts['answered_correctly_sum_b_dict'][u]
        cb[cnt] = pdicts['count_b_dict'][u]
        expacsu[cnt] = pdicts['pexp_answered_correctly_sum_u_dict'][u]
        expcu[cnt] = pdicts['pexp_count_u_dict'][u]
        cidacsu[cnt] = pdicts['content_id_answered_correctly_sum_u_dict'][ucid]
        cidcu[cnt] = pdicts['content_id_count_u_dict'][ucid]
        tstamp[cnt] = [tstmp - pdicts[f'lag_time{i}'][u] for i in [0,1,2,4,9]]
        tstavg[cnt] = pdicts['lag_time_avg'][u]/ (pdicts['count_u_dict'][u]+0.1)
        qamat[cnt] = pdicts['qaRankcum'][u] / (pdicts['count_u_dict'][u] + 0.01), \
                pdicts['qaRatiocum'][u] / (pdicts['count_u_dict'][u] + 0.01), \
                pdicts['qaRankcum'][u] / (pdicts['qaRankCorrectcum'][u] + 0.01), \
                pdicts['qaRatiocum'][u] / (pdicts['qaRatioCorrectcum'][u] + 0.01), \
                pdicts['qaRankFirstcum'][u] / (pdicts['qaRankCorrectFirstcum'][u] + 0.01), \
                pdicts['qaRatioFirstcum'][u] / (pdicts['qaRatioCorrectFirstcum'][u] + 0.01)
        
        partsdict[part]['acsu'][cnt]  = pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        partsdict[part]['cu'][cnt] = pdicts[f'{part}p_count_u_dict'][u]
        
        for p in range(1,8):
            if p == part: continue
            partsdict[p]['cu'][cnt] = pdicts[f'{p}p_count_u_dict'][u]
            partsdict[p]['acsu'][cnt]  = pdicts[f'{p}p_answered_correctly_sum_u_dict'][u]

            
        if update:
            pdicts['count_u_dict'][u] += 1
            lag0daysdecay = (max(30, (tstmp - pdicts[f'lag_time0'][u])/(3600*24*1000))) / 60
            pdicts['count_u_decay_dict'][u] = 1 + (1-lag0daysdecay) * pdicts['count_u_decay_dict'][u]
            pdicts['answered_correctly_sum_u_decay_dict'][u] = yprev +  \
                                    (1-lag0daysdecay) * pdicts['answered_correctly_sum_u_decay_dict'][u]
            lag0daysdecay = (max(100, (tstmp - pdicts[f'lag_time0'][u])/(3600*24*1000))) / 100
            pdicts['count_u_decay_dict1'][u] = 1 + (1-lag0daysdecay) * pdicts['count_u_decay_dict1'][u]
            pdicts['answered_correctly_sum_u_decay_dict1'][u] = yprev +  \
                                    (1-lag0daysdecay) * pdicts['answered_correctly_sum_u_decay_dict1'][u]
            try:
                pdicts['qaRankCorrectcum'][u] += pdicts['qaRank'][(cid, pdicts['qaCorrect'][cid])]
                pdicts['qaRatioCorrectcum'][u] += pdicts['qaRatio'][(cid, pdicts['qaCorrect'][cid])]
                pdicts['userAvgRatioCum'][u] += pdicts['qaRatio'][(cid, pdicts['qaCorrect'][cid])]
                pdicts['userRatioCum'][u] += yprev
            except:
                pdicts['qaRankCorrectcum'][u] += 0
                pdicts['qaRatioCorrectcum'][u] += 1.
            try:
                pdicts['qaRankcum'][u] += pdicts['qaRank'][(cid, ua)]
                pdicts['qaRatiocum'][u] += pdicts['qaRatio'][(cid, ua)]
            except:
                pdicts['qaRankcum'][u] += 4.
                pdicts['qaRatiocum'][u] += 0.1
            
            # Only add this on first attempts, and track first attempt success rate
            if pdicts['content_id_answered_correctly_sum_u_dict'][ucid] == 0:
                try:
                    pdicts['qaRankCorrectFirstcum'][u] += pdicts['qaRankFirst'][(cid, pdicts['qaCorrect'][cid])]
                    pdicts['qaRatioCorrectFirstcum'][u] += pdicts['qaRatioFirst'][(cid, pdicts['qaCorrect'][cid])]
                except:
                    pdicts['qaRankCorrectFirstcum'][u] += 0
                    pdicts['qaRatioCorrectFirstcum'][u] += 1.
                try:
                    pdicts['qaRankFirstcum'][u] += pdicts['qaRankFirst'][(cid, ua)]
                    pdicts['qaRatioFirstcum'][u] += pdicts['qaRatioFirst'][(cid, ua)]
                except:
                    pdicts['qaRankFirstcum'][u] += 4.
                    pdicts['qaRatioFirstcum'][u] += 0.1
                
            #pdicts['ctunique'][uiddict[u]].add(cid)
            
            for i in list(range(1, 10))[::-1]:
                pdicts[f'lag_time{i}'][u] = pdicts[f'lag_time{i-1}'][u] 
            pdicts['lag_time0'][u] = tstmp
            pdicts['lag_time_avg'][u] += tstmp
            pdicts['count_c_dict'][cid] += 1
            pdicts[f'{part}p_count_u_dict'][u] += 1
            pdicts['content_id_count_u_dict'][ucid] += 1
            pdicts['count_b_dict'][u] = 1 if newbid else pdicts['count_b_dict'][u] + 1
            if newbid : pdicts['answered_correctly_sum_b_dict'][u] = 0
            if yprev: 
                pdicts['answered_correctly_sum_u_dict'][u] += 1
                pdicts['answered_correctly_sum_c_dict'][cid] += 1
                pdicts['answered_correctly_sum_b_dict'][u] += 1
                pdicts['content_id_answered_correctly_sum_u_dict'][ucid] += 1
                pdicts[f'{part}p_answered_correctly_sum_u_dict'][u] += 1
            if pexp:
                if yprev: 
                    pdicts['pexp_answered_correctly_sum_u_dict'][u] += yprev
                pdicts['pexp_count_u_dict'][u] += 1
            pdicts['track_b'][u] = bid
            
            
                
    for t1, (matcu, matascu) in enumerate(zip([cu, expcu, cidcu, cb], [acsu, expacsu, cidacsu, acsb])):
        df[f'counts___feat{t1}'] = matcu
        df[f'avgcorrect___feat{t1}'] =  (matascu / (matcu + 0.001)).astype(np.float16)
        #gc.collect()
    df['cid_answered_correctly'] = acsu
    df[[f'lag_content_time{i}' for i in [0,1,2,5,10]]] = tstamp
    df['lag_content_avgtime'] = tstavg
    #df[['ctunique_sum', 'ctunique_attempt_ration']] = ctunq
    df[[f'lecture_stats_{i}' for i in range(7)]] = lectct
    df[[f'lecture_stats_{i}' for i in range(7,10)]] = lectavg
    df[[f'rank_stats_{i}' for i in range(6)]] = qamat
    df[[f'nattempt_lag_{i}' for i in range(4)]] = nattmpt
    df[[f'bundle_time_ratio{i}' for i in range(3)]] = bratio.clip(0,4)
    df[[f'decayed_avg_correct{i}' for i in range(2)]] = acsudec
    
    
    del cu, expcu, acsu, expacsu
    for t, i in enumerate(range(1,8)):  
        df[f'counts___feat{t1+t+1}'] = partsdict[i]['cu']
        df[f'avgcorrect___feat{t1+t+1}'] =  (partsdict[i]['acsu']  / (partsdict[i]['cu'] + 0.001)).astype(np.float16)
        del partsdict[i]
    
    return df

def add_user_feats_without_update(df, pdicts, update=False):
    df = add_user_feats(df, pdicts, update)
    return df

def update_user_feats(df, pdicts):
    filtcols = ['user_id','answered_correctly', 'part', 'prior_question_had_explanation', 'content_type_id']

    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for row in df[filtcols].fillna(False).values:
        u, yprev, part, pexp, ctype = row
        if ctype == 0:
            pdicts['count_u_dict'][u] += 1
            pdicts[f'{part}p_count_u_dict'][u] += 1
            if yprev: 
                pdicts['answered_correctly_sum_u_dict'][u] += 1
                pdicts[f'{part}p_answered_correctly_sum_u_dict'][u] += 1
            if pexp:
                if yprev: 
                    pdicts['pexp_answered_correctly_sum_u_dict'][u] += yprev
                pdicts['pexp_count_u_dict'][u] += 1

CUT=0
DIR='val'
VERSION='V1'
debug = False
validaten_flg = False
FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', \
                       'timestamp', 'user_answer']
valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]

train.query('user_id == 1857133602')[['user_id', 'content_type_id', 'prior_question_elapsed_time']].head(1000)

# Treat tags same as content, but but just loop update 
# Carry the 

questions_df = pd.read_csv('data/questions.csv')
ldf = pd.read_csv('data/lectures.csv')
ldf.type_of = ldf.type_of.str.replace(' ', '_')
ldict = ldf.set_index('lecture_id').to_dict()
lecture_types = [t for t in ldf.type_of.unique() if t!= 'starter']


bdict = questions_df.set_index('question_id')['bundle_id'].to_dict()
keepcols = ['question_id', 'part', 'tags', 'bundle_id', 'correct_answer']
train = pd.merge(train, questions_df[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
formatcols =  ['question_id', 'part', 'bundle_id', 'correct_answer', 'user_answer']
train[formatcols] = train[formatcols].fillna(0).astype(np.int16)
valid[formatcols] = valid[formatcols].fillna(0).astype(np.int16)

ix = train.content_type_id == False
bdf = train[ix][['user_id', 'timestamp', 'bundle_id', 'prior_question_elapsed_time']].sort_values(['user_id', 'timestamp'])
bdf = pd.concat([bdf, bdf.shift(1).add_suffix('_lag')], 1)
bdf = bdf.dropna(how='any').query('user_id == user_id_lag')
bdf = bdf.groupby(['user_id', 'timestamp', 'bundle_id_lag'])['prior_question_elapsed_time'].mean()
bdf = bdf.reset_index().dropna(how='any')
bprevdict = bdf.groupby('bundle_id_lag')['prior_question_elapsed_time'].median().to_dict()

# How correct is the answer
def qaRanks(df):
    aggdf1 = df.groupby(['question_id', 'user_answer', 'correct_answer'])['answered_correctly'].count()
    aggdf2 = df.groupby(['question_id'])['answered_correctly'].count()
    aggdf = pd.merge(aggdf1, aggdf2, left_index=True, right_index = True).reset_index()
    aggdf.columns = ['question_id', 'user_answer', 'correct_answer', 'answcount', 'quescount']
    aggdf['answerratio'] = (aggdf.answcount / aggdf.quescount).astype(np.float16)
    aggdf['answerrank'] = aggdf.groupby("question_id")["answerratio"].rank("dense", ascending=False).astype(np.int8)
    rankDf = aggdf.set_index('question_id')[['answerrank', 'answerratio', 'answcount']].reset_index()
    qaRank = aggdf.set_index(['question_id', 'user_answer']).answerrank.to_dict()
    qaRatio = aggdf.set_index(['question_id', 'user_answer']).answerratio.to_dict()
    qaCorrect = questions_df.set_index('question_id').correct_answer.to_dict()
    return qaRank, qaRatio, qaCorrect, rankDf

ix = train.content_type_id == False
qaRank, qaRatio, qaCorrect, rankDf = qaRanks(train[ix])
answerrank = train.groupby(["user_id", "content_id"])["timestamp"]\
                    .rank("dense", ascending=True).astype(np.int8)
qaRankFirst, qaRatioFirst, _, _ = qaRanks(train[ix][answerrank==1])

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

content_df1 = train.query('content_type_id == 0')[['content_id','answered_correctly']]\
                .groupby(['content_id']).agg(['mean', 'count']).astype(np.float16).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c', 'answered_correctly_ct_c']
content_df2 = train.query('content_type_id == 0') \
                .groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float16).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df3 = train.query('content_type_id == 0')[['content_id', 'user_id','answered_correctly']] \
                .drop_duplicates(keep='first')
content_df3 = content_df3[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df3.columns = ['content_id', 'answered_correctly_first_avg_c']
content_df4 = train.query('content_type_id == 0')[['content_id', 'user_id','answered_correctly']] \
                .drop_duplicates(keep='last')
content_df4 = content_df4[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df4.columns = ['content_id', 'answered_correctly_last_avg_c']

content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df  = pd.merge(content_df, content_df3, on = 'content_id')
content_df  = pd.merge(content_df, content_df4, on = 'content_id')
content_df.columns
del content_df1, content_df2, content_df3, content_df4
gc.collect()

content_df.iloc[:,1:] = content_df.iloc[:,1:].astype(np.float16)
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

# tags
train.tags = train.tags.fillna('')
valid.tags = valid.tags.fillna('')
args = {'tokenizer':None, 'stop_words':None, 'strip_accents':None}
'''
if False:
    vec = CountVectorizer(**args)
    tags_as_matrix = vec.fit_transform(train[:2*10**6].tags.tolist())
    taglda = LDA(n_components=5, n_jobs=4, verbose = 1).fit(tags_as_matrix)
    dumpobj(f'data/val/vectoriser_{VERSION}.pk', vec)
    dumpobj(f'data/val/tag_lda_{VERSION}.pk', taglda)
else:
    vec = loadobj(f'data/val/vectoriser_{VERSION}.pk')
    taglda = loadobj(f'data/val/tag_lda_{VERSION}.pk')

if False:
    train[[f'lda_comp{i}' for i in range(5)]] = taglda.transform(vec.transform(train.tags.tolist()))
    valid[[f'lda_comp{i}' for i in range(5)]] = taglda.transform(vec.transform(valid.tags.tolist()))
    dumpobj(f'data/val/trainlda_{VERSION}.pk', train[[f'lda_comp{i}' for i in range(5)]])
    dumpobj(f'data/val/validlda_{VERSION}.pk', valid[[f'lda_comp{i}' for i in range(5)]])
else:
    train[[f'lda_comp{i}' for i in range(5)]] = loadobj(f'data/val/trainlda_{VERSION}.pk')
    valid[[f'lda_comp{i}' for i in range(5)]] = loadobj(f'data/val/validlda_{VERSION}.pk')
'''
#uiddict = dict((u,t) for t,u in \
#               enumerate(set(train.user_id.unique().tolist()+valid.user_id.unique().tolist())))
#nuser = max([k for k in uiddict.values()])

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
          'tag_answered_correctly_sum_u_dict' : defaultdict(int),
          'tag_count_u_dict': defaultdict(int),
          'lag_time_avg': defaultdict(int),
          'cum_lecture_ct' : defaultdict(int),
          'cum_lecture_time' : defaultdict(int),
          'container_id_u': defaultdict(int),
          'container_id_u_ct': defaultdict(int),
          'prev_lecture_part' : defaultdict(int),
          'prev_lecture_time' : defaultdict(int),
          'prev_lecture_timestamp' : defaultdict(int),
          'prev_lecture_timestamp2' : defaultdict(int),
          'prev_lecture_timestamp3' : defaultdict(int),
          'prior_explanation2' : defaultdict(int),
          'prior_explanation3' : defaultdict(int),
          'cum_ques_since_lecture' : defaultdict(int),
          'ques_reattempt_count_lag2' : defaultdict(int),
          'ques_reattempt_count_lag1' : defaultdict(int),
          'ques_reattempt_count' : defaultdict(int),
          'container_reattempt_count_lag1' : defaultdict(int),
          'container_reattempt_count' : defaultdict(int),
          'container_ques': defaultdict(int),
          'container_curr' : defaultdict(int),
          'container_unq_ques_lag1' : defaultdict(int),
          'container_unq_ques' : defaultdict(set),
          'container_lag0' : defaultdict(int),
          'container_lag1' : defaultdict(int),
          'container_lag2' : defaultdict(int),
          #'ctunique' : [set() for i in range(nuser+1)],
          #'uiddict' : uiddict,
          'bundprevtime' : bprevdict,
          'bundprev' : defaultdict(int),
          'bundcurr' : defaultdict(int),
          'bundpathcum' : defaultdict(int),
          'bundusercum' : defaultdict(int),
          'userAvgRatioCum': defaultdict(float),
          'userRatioCum' : defaultdict(int),
          'answered_correctly_sum_u_decay_dict': defaultdict(float),
          'count_u_decay_dict': defaultdict(float),
          
          'qaRank' : qaRank,
          'qaRatio' : qaRatio,
          'qaRankFirst' : qaRankFirst, 
          'qaRatioFirst' : qaRatioFirst, 
          'qaCorrect': qaCorrect,
          'qaRankcum' : defaultdict(int),
          'qaRatiocum' : defaultdict(int),
          'qaRankFirstcum' : defaultdict(int),
          'qaRatioFirstcum' : defaultdict(int),
          'qaRankCorrectcum' : defaultdict(int),
          'qaRatioCorrectcum' : defaultdict(int),
          'qaRankCorrectFirstcum' : defaultdict(int),
          'qaRatioCorrectFirstcum' : defaultdict(int),
          'content_id_lag' : defaultdict(int), 
          'pexp_answered_correctly_sum_u_dict' : defaultdict(int),
          'pexp_count_u_dict': defaultdict(int)}

for i in range(10): 
    pdicts[f'lag_time{i}'] =   defaultdict(int)
for p in train.part.unique():
    pdicts[f'{p}p_answered_correctly_sum_u_dict'] =  defaultdict(int)
    pdicts[f'{p}p_count_u_dict'] =  defaultdict(int)
    pdicts[f'{p}l_count_u_dict'] =  defaultdict(int)

train = add_user_feats(train, pdicts)
valid = add_user_feats(valid, pdicts)


train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

train[[f'tag{i}' for i in range(6)]] = \
    train.tags.apply(lambda x: list(map(int, x.split(' ')))+[188]*(6-len(x.split(' '))) ).tolist()
valid[[f'tag{i}' for i in range(6)]] = \
    valid.tags.apply(lambda x: list(map(int, x.split(' ')))+[188]*(6-len(x.split(' '))) ).tolist()


# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# answered correctly average for each content
# use only last 30M training data for limited memory on kaggle env.
#train = train[-30000000:]
train[[f'rank_stats_diff_{i}' for i in [0,1]]] = (train[[f'rank_stats_{i}' for i in [0,3]]].values - \
                                            train[[f'rank_stats_{i}' for i in [2,1]]].values)

valid[[f'rank_stats_diff_{i}' for i in [0,1]]] = (valid[[f'rank_stats_{i}' for i in [0,3]]].values - \
                                            valid[[f'rank_stats_{i}' for i in [2,1]]].values)
'''
enc = set(train.user_id.value_counts()[:2000].index.tolist() +\
    valid.user_id.value_counts()[:1000].index.tolist())
usermap = dict((u, t+1) for t,u in enumerate(enc))
usermap = defaultdict(int, usermap)

valid['user_id_enc'] = valid.user_id.map(usermap)
train['user_id_enc'] = train.user_id.map(usermap)
'''


'''
ix = train.content_type_id == False
df = train[ix][['user_id', 'timestamp', 'bundle_id', 'prior_question_elapsed_time']].sort_values(['user_id', 'timestamp'])
df = pd.concat([df, df.shift(1).add_suffix('_lag')], 1)
df = df.dropna(how='any').query('user_id == user_id_lag')
df = df.groupby(['user_id', 'timestamp', 'bundle_id_lag'])['prior_question_elapsed_time'].mean()
df = df.reset_index().dropna(how='any')
tprevdict = df.groupby('bundle_id_lag')['prior_question_elapsed_time'].median().to_dict()
'''

#train['part_pexp'] = (2*(train['part'] + 0.5 * train['prior_question_had_explanation'])).astype(np.float32)
#valid['part_pexp'] = (2*(valid['part'] + 0.5 * valid['prior_question_had_explanation'])).astype(np.float32)
train['content_id_pexp'] = (2*(train['content_id'] + 0.5 * train['prior_question_had_explanation'])).astype(np.float32)
valid['content_id_pexp'] = (2*(valid['content_id'] + 0.5 * valid['prior_question_had_explanation'])).astype(np.float32)


TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', 'cid_answered_correctly', \
         'content_id', 'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', 'correct_answer', \
         'answered_correctly_ct_c', 'answered_correctly_last_avg_c', 'lag_content_avgtime', \
         ]
#FEATS = ['cid_answered_correctly', 'content_id', 'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', \
#         'lag_content_avgtime']
FEATS += [f'counts___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
FEATS += [f'lag_content_time{i}' for i in [0,1,2,5,10]]
FEATS += [f'lecture_stats_{i}' for i in range(10)]
FEATS += [f'tag{i}' for i in range(6)]
FEATS += [f'rank_stats_{i}' for i in [1,3,5]]
FEATS += [f'rank_stats_diff_{i}' for i in [1]]
FEATS += [f'nattempt_lag_{i}' for i in range(4)]
FEATS += [f'bundle_time_ratio{i}' for i in range(3)]
FEATS += [f'decayed_avg_correct{i}' for i in range(2)]
y_tr = train[TARGET]
y_va = valid[TARGET]
_=gc.collect()

categoricals = ['part', 'content_id'] + [f'tag{i}' for i in range(6)]
lgb_train = lgb.Dataset(train[FEATS], y_tr, categorical_feature = categoricals)
lgb_valid = lgb.Dataset(valid[FEATS], y_va, categorical_feature = categoricals)
_=gc.collect()

model = lgb.train(
                    {'objective': 'binary', 
                     'min_data_per_group':5000,
                     'min_data_in_leaf' : 300,
                     'learning_rate': 0.1},
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    early_stopping_rounds=60,
                )
print('auc:', roc_auc_score(y_va.values, model.predict(valid[FEATS])  )  )
_ = lgb.plot_importance(model, max_num_features = 64, figsize = (5,15))

#### Try 
#- Any way to do nunique questions - like a boolean array / sparse array
#- average sucess on second, third try

# For curr ques, prev ques, prev response, proba right

#### New 
#- 300 min_leaf
# Some container lag feature
# Cumulative prev time vs global prev time
# correct anwser as raw feature
# timestamp as raw feature
# Rolling ratio of correct answers vs average correct
'''
Training until validation scores don't improve for 60 rounds
[100]	training's binary_logloss: 0.520154	valid_1's binary_logloss: 0.534322
[200]	training's binary_logloss: 0.515646	valid_1's binary_logloss: 0.531363
[300]	training's binary_logloss: 0.51313	valid_1's binary_logloss: 0.529889
[400]	training's binary_logloss: 0.511403	valid_1's binary_logloss: 0.529003
[500]	training's binary_logloss: 0.510012	valid_1's binary_logloss: 0.528415
[600]	training's binary_logloss: 0.50887	valid_1's binary_logloss: 0.527922
[700]	training's binary_logloss: 0.507957	valid_1's binary_logloss: 0.527655
[800]	training's binary_logloss: 0.507122	valid_1's binary_logloss: 0.527367
[900]	training's binary_logloss: 0.506293	valid_1's binary_logloss: 0.52709
[1000]	training's binary_logloss: 0.505543	valid_1's binary_logloss: 0.526856
[1100]	training's binary_logloss: 0.504794	valid_1's binary_logloss: 0.52662
[1200]	training's binary_logloss: 0.504259	valid_1's binary_logloss: 0.526527
[1300]	training's binary_logloss: 0.50371	valid_1's binary_logloss: 0.526407
[1400]	training's binary_logloss: 0.503151	valid_1's binary_logloss: 0.526306
[1500]	training's binary_logloss: 0.502634	valid_1's binary_logloss: 0.526219
[1600]	training's binary_logloss: 0.502024	valid_1's binary_logloss: 0.526122
[1700]	training's binary_logloss: 0.501527	valid_1's binary_logloss: 0.526046
[1800]	training's binary_logloss: 0.501105	valid_1's binary_logloss: 0.526037
[1900]	training's binary_logloss: 0.50054	valid_1's binary_logloss: 0.525943
[2000]	training's binary_logloss: 0.500156	valid_1's binary_logloss: 0.525908
Early stopping, best iteration is:
[1984]	training's binary_logloss: 0.500203	valid_1's binary_logloss: 0.525904
auc: 0.7914469566241847
'''


'''
Training until validation scores don't improve for 60 rounds
[100]	training's binary_logloss: 0.520439	valid_1's binary_logloss: 0.534778
[200]	training's binary_logloss: 0.515963	valid_1's binary_logloss: 0.531718
[300]	training's binary_logloss: 0.513414	valid_1's binary_logloss: 0.530255
[400]	training's binary_logloss: 0.511694	valid_1's binary_logloss: 0.529395
[500]	training's binary_logloss: 0.510424	valid_1's binary_logloss: 0.528798
[600]	training's binary_logloss: 0.509297	valid_1's binary_logloss: 0.528341
[700]	training's binary_logloss: 0.508402	valid_1's binary_logloss: 0.527997
[800]	training's binary_logloss: 0.507574	valid_1's binary_logloss: 0.527771
[900]	training's binary_logloss: 0.506814	valid_1's binary_logloss: 0.527564
[1000]	training's binary_logloss: 0.506165	valid_1's binary_logloss: 0.527362
[1100]	training's binary_logloss: 0.505597	valid_1's binary_logloss: 0.527254
[1200]	training's binary_logloss: 0.505054	valid_1's binary_logloss: 0.527119
[1300]	training's binary_logloss: 0.504465	valid_1's binary_logloss: 0.527025
[1400]	training's binary_logloss: 0.503897	valid_1's binary_logloss: 0.52696
[1500]	training's binary_logloss: 0.503539	valid_1's binary_logloss: 0.52691
[1600]	training's binary_logloss: 0.502938	valid_1's binary_logloss: 0.526807
[1700]	training's binary_logloss: 0.502468	valid_1's binary_logloss: 0.526745
[1800]	training's binary_logloss: 0.501954	valid_1's binary_logloss: 0.526682
Early stopping, best iteration is:
[1784]	training's binary_logloss: 0.502017	valid_1's binary_logloss: 0.526676
auc: 0.7906527768045082
'''


'''
Adding new containers
Training until validation scores don't improve for 40 rounds
[100]	training's binary_logloss: 0.521018	valid_1's binary_logloss: 0.535465
[200]	training's binary_logloss: 0.516912	valid_1's binary_logloss: 0.532599
[300]	training's binary_logloss: 0.514828	valid_1's binary_logloss: 0.531332
[400]	training's binary_logloss: 0.513335	valid_1's binary_logloss: 0.530567
[500]	training's binary_logloss: 0.512002	valid_1's binary_logloss: 0.529895
[600]	training's binary_logloss: 0.511015	valid_1's binary_logloss: 0.529518
[700]	training's binary_logloss: 0.510074	valid_1's binary_logloss: 0.529082
[800]	training's binary_logloss: 0.50921	valid_1's binary_logloss: 0.528789
[900]	training's binary_logloss: 0.508436	valid_1's binary_logloss: 0.528559
[1000]	training's binary_logloss: 0.507828	valid_1's binary_logloss: 0.528427
[1100]	training's binary_logloss: 0.507204	valid_1's binary_logloss: 0.528289
[1200]	training's binary_logloss: 0.506684	valid_1's binary_logloss: 0.528165
[1300]	training's binary_logloss: 0.506021	valid_1's binary_logloss: 0.528018
[1400]	training's binary_logloss: 0.505465	valid_1's binary_logloss: 0.527912
[1500]	training's binary_logloss: 0.505002	valid_1's binary_logloss: 0.527834
[1600]	training's binary_logloss: 0.504488	valid_1's binary_logloss: 0.527727
[1700]	training's binary_logloss: 0.503974	valid_1's binary_logloss: 0.52763
[1800]	training's binary_logloss: 0.503434	valid_1's binary_logloss: 0.527505
[1900]	training's binary_logloss: 0.502941	valid_1's binary_logloss: 0.527442
[2000]	training's binary_logloss: 0.502479	valid_1's binary_logloss: 0.527384
[2100]	training's binary_logloss: 0.501954	valid_1's binary_logloss: 0.527326
Early stopping, best iteration is:
[2138]	training's binary_logloss: 0.501707	valid_1's binary_logloss: 0.527262
auc: 0.790103963188252
'''

'''
# Remove ct unique for memory
Training until validation scores don't improve for 40 rounds
[100]	training's binary_logloss: 0.522005	valid_1's binary_logloss: 0.536513
[200]	training's binary_logloss: 0.517841	valid_1's binary_logloss: 0.533511
[300]	training's binary_logloss: 0.515728	valid_1's binary_logloss: 0.532336
[400]	training's binary_logloss: 0.514141	valid_1's binary_logloss: 0.531461
[500]	training's binary_logloss: 0.512944	valid_1's binary_logloss: 0.530925
[600]	training's binary_logloss: 0.51178	valid_1's binary_logloss: 0.530449
[700]	training's binary_logloss: 0.51086	valid_1's binary_logloss: 0.530171
[800]	training's binary_logloss: 0.510065	valid_1's binary_logloss: 0.529965
[900]	training's binary_logloss: 0.509396	valid_1's binary_logloss: 0.529809
[1000]	training's binary_logloss: 0.508774	valid_1's binary_logloss: 0.529632
[1100]	training's binary_logloss: 0.508176	valid_1's binary_logloss: 0.52949
[1200]	training's binary_logloss: 0.507474	valid_1's binary_logloss: 0.529281
[1300]	training's binary_logloss: 0.506791	valid_1's binary_logloss: 0.529126
[1400]	training's binary_logloss: 0.506311	valid_1's binary_logloss: 0.529
Early stopping, best iteration is:
[1405]	training's binary_logloss: 0.506269	valid_1's binary_logloss: 0.52899
auc: 0.788436261707026

auc: 0.7891421361249153 (with LR 0.03)
'''

'''
Training until validation scores don't improve for 40 rounds
[100]	training's binary_logloss: 0.521941	valid_1's binary_logloss: 0.536619
[200]	training's binary_logloss: 0.517715	valid_1's binary_logloss: 0.533693
[300]	training's binary_logloss: 0.515373	valid_1's binary_logloss: 0.532383
[400]	training's binary_logloss: 0.513757	valid_1's binary_logloss: 0.531588
[500]	training's binary_logloss: 0.512462	valid_1's binary_logloss: 0.530979
[600]	training's binary_logloss: 0.511447	valid_1's binary_logloss: 0.530575
[700]	training's binary_logloss: 0.510686	valid_1's binary_logloss: 0.530361
[800]	training's binary_logloss: 0.509906	valid_1's binary_logloss: 0.530124
[900]	training's binary_logloss: 0.509147	valid_1's binary_logloss: 0.529869
[1000]	training's binary_logloss: 0.508407	valid_1's binary_logloss: 0.529674
[1100]	training's binary_logloss: 0.507772	valid_1's binary_logloss: 0.529575
[1200]	training's binary_logloss: 0.50696	valid_1's binary_logloss: 0.52935
[1300]	training's binary_logloss: 0.506307	valid_1's binary_logloss: 0.529213
[1400]	training's binary_logloss: 0.505698	valid_1's binary_logloss: 0.529065
[1500]	training's binary_logloss: 0.505216	valid_1's binary_logloss: 0.528985
[1600]	training's binary_logloss: 0.504641	valid_1's binary_logloss: 0.528929
Early stopping, best iteration is:
[1645]	training's binary_logloss: 0.504318	valid_1's binary_logloss: 0.528858
auc: 0.7885549069565729
'''


'''
Training until validation scores don't improve for 40 rounds
[100]	training's binary_logloss: 0.525822	valid_1's binary_logloss: 0.540236
[200]	training's binary_logloss: 0.521471	valid_1's binary_logloss: 0.537077
[300]	training's binary_logloss: 0.519208	valid_1's binary_logloss: 0.535764
[400]	training's binary_logloss: 0.517621	valid_1's binary_logloss: 0.534939
[500]	training's binary_logloss: 0.516439	valid_1's binary_logloss: 0.534419
[600]	training's binary_logloss: 0.51525	valid_1's binary_logloss: 0.534065
[700]	training's binary_logloss: 0.514439	valid_1's binary_logloss: 0.533809
[800]	training's binary_logloss: 0.513546	valid_1's binary_logloss: 0.533493
[900]	training's binary_logloss: 0.512978	valid_1's binary_logloss: 0.533405
[1000]	training's binary_logloss: 0.512349	valid_1's binary_logloss: 0.533302
[1100]	training's binary_logloss: 0.511613	valid_1's binary_logloss: 0.533173
[1200]	training's binary_logloss: 0.510918	valid_1's binary_logloss: 0.533039
[1300]	training's binary_logloss: 0.510449	valid_1's binary_logloss: 0.532971
[1400]	training's binary_logloss: 0.509825	valid_1's binary_logloss: 0.532857
Early stopping, best iteration is:
[1393]	training's binary_logloss: 0.509846	valid_1's binary_logloss: 0.532853
auc: 0.784237
'''

'''
Training until validation scores don't improve for 40 rounds
[100]	training's binary_logloss: 0.526205	valid_1's binary_logloss: 0.54046
[200]	training's binary_logloss: 0.522604	valid_1's binary_logloss: 0.537668
[300]	training's binary_logloss: 0.520766	valid_1's binary_logloss: 0.536486
[400]	training's binary_logloss: 0.519574	valid_1's binary_logloss: 0.535862
[500]	training's binary_logloss: 0.518494	valid_1's binary_logloss: 0.53539
[600]	training's binary_logloss: 0.517464	valid_1's binary_logloss: 0.534978
[700]	training's binary_logloss: 0.516631	valid_1's binary_logloss: 0.534704
[800]	training's binary_logloss: 0.515858	valid_1's binary_logloss: 0.534483
[900]	training's binary_logloss: 0.515113	valid_1's binary_logloss: 0.534271
[1000]	training's binary_logloss: 0.514348	valid_1's binary_logloss: 0.534024
[1100]	training's binary_logloss: 0.513675	valid_1's binary_logloss: 0.533898
Early stopping, best iteration is:
[1104]	training's binary_logloss: 0.513646	valid_1's binary_logloss: 0.533886
auc: 0.7832431666460757
'''

'''
Training until validation scores don't improve for 20 rounds
[100]	training's binary_logloss: 0.528732	valid_1's binary_logloss: 0.542389
[200]	training's binary_logloss: 0.526037	valid_1's binary_logloss: 0.539961
[300]	training's binary_logloss: 0.524654	valid_1's binary_logloss: 0.538942
[400]	training's binary_logloss: 0.523694	valid_1's binary_logloss: 0.538419
[500]	training's binary_logloss: 0.522916	valid_1's binary_logloss: 0.538019
[600]	training's binary_logloss: 0.522205	valid_1's binary_logloss: 0.537713
[700]	training's binary_logloss: 0.521619	valid_1's binary_logloss: 0.537492
[800]	training's binary_logloss: 0.521093	valid_1's binary_logloss: 0.53735
[900]	training's binary_logloss: 0.520584	valid_1's binary_logloss: 0.537127
[1000]	training's binary_logloss: 0.520076	valid_1's binary_logloss: 0.536953
[1100]	training's binary_logloss: 0.519589	valid_1's binary_logloss: 0.536801
Early stopping, best iteration is:
[1156]	training's binary_logloss: 0.519335	valid_1's binary_logloss: 0.536728
auc: 0.7800652346074616
'''


'''
# Without tags
Training until validation scores don't improve for 20 rounds
[100]	training's binary_logloss: 0.531441	valid_1's binary_logloss: 0.545613
[200]	training's binary_logloss: 0.529406	valid_1's binary_logloss: 0.543914
[300]	training's binary_logloss: 0.528288	valid_1's binary_logloss: 0.543117
[400]	training's binary_logloss: 0.527503	valid_1's binary_logloss: 0.542622
[500]	training's binary_logloss: 0.526878	valid_1's binary_logloss: 0.542292
Early stopping, best iteration is:
[578]	training's binary_logloss: 0.526435	valid_1's binary_logloss: 0.542062
auc: 0.7740469062728539
'''

'''
[100]	training's binary_logloss: 0.536142	valid_1's binary_logloss: 0.5509
[200]	training's binary_logloss: 0.534503	valid_1's binary_logloss: 0.549571
[300]	training's binary_logloss: 0.533607	valid_1's binary_logloss: 0.548998
[400]	training's binary_logloss: 0.532981	valid_1's binary_logloss: 0.548699
[500]	training's binary_logloss: 0.532449	valid_1's binary_logloss: 0.548394
[600]	training's binary_logloss: 0.531999	valid_1's binary_logloss: 0.548271
[700]	training's binary_logloss: 0.531576	valid_1's binary_logloss: 0.548121
Early stopping, best iteration is:
[767]	training's binary_logloss: 0.531288	valid_1's binary_logloss: 0.548
auc: 0.7673362503126356
'''

'''
[100]	training's binary_logloss: 0.536789	valid_1's binary_logloss: 0.551813
[200]	training's binary_logloss: 0.535228	valid_1's binary_logloss: 0.550596
[300]	training's binary_logloss: 0.53441	valid_1's binary_logloss: 0.550027
[400]	training's binary_logloss: 0.533811	valid_1's binary_logloss: 0.549708
[500]	training's binary_logloss: 0.533319	valid_1's binary_logloss: 0.549507
[600]	training's binary_logloss: 0.532879	valid_1's binary_logloss: 0.549331
[700]	training's binary_logloss: 0.5325	valid_1's binary_logloss: 0.549199
[800]	training's binary_logloss: 0.532134	valid_1's binary_logloss: 0.549072
[900]	training's binary_logloss: 0.531775	valid_1's binary_logloss: 0.548953
[1000]	training's binary_logloss: 0.531456	valid_1's binary_logloss: 0.548855
[1100]	training's binary_logloss: 0.531137	valid_1's binary_logloss: 0.548796
Early stopping, best iteration is:
[1165]	training's binary_logloss: 0.530921	valid_1's binary_logloss: 0.548744
auc: 0.7665125853545588
'''

