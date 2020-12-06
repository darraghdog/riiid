# https://www.kaggle.com/its7171/lgbm-with-loop-feature-engineering/#data
import os
import sys
os.chdir('/data/riiid')
sys.path.insert(0, '/data/riiid')
#os.chdir('/Users/dhanley/Documents/riiid/')
import sys
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import lightgbm as lgb
import warnings
from scripts.utils import Iter_Valid, dumpobj, loadobj
from sklearn.externals import joblib
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# user stats features with loops
def getKeys(train):
    kd = {'userCtr': 1, 
                'usercontentCtr': 1,
                'userCtr': 1,
                'usercontentKey' : defaultdict(lambda: {}),
                'userKey' : {}}#,
    for row in tqdm(train[['user_id', 'content_id']].values):
        user, cont = row 
        if cont not in kd['usercontentKey'][user]:
            kd['usercontentKey'][user][cont] = kd['usercontentCtr']
            kd['usercontentCtr'] += 1
        if user not in kd['userKey']:
            kd['userKey'][user] = kd['userCtr']
            kd['userCtr'] += 1
    return kd


# funcs for user stats with loop
def add_user_feats(df, pdicts, kdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
    acsudec = np.zeros((len(df),2), dtype=np.float16)
    cu = np.zeros(len(df), dtype=np.uint32)
    pcu = np.zeros(len(df), dtype=np.uint32)
    acsb = np.zeros(len(df), dtype=np.uint32)
    cb = np.zeros(len(df), dtype=np.uint32)
    expacsu = np.zeros(len(df), dtype=np.uint32)
    expcu = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    tcidacsu = np.zeros(len(df), dtype=np.uint32)
    tcidcu = np.zeros(len(df), dtype=np.uint32)
    tstamp = np.zeros((len(df), 5), dtype=np.uint32)
    tstavg = np.zeros(len(df), dtype=np.float32)
    lectct = np.zeros((len(df), 7), dtype=np.uint32)
    lectavg = np.zeros((len(df), 3), dtype=np.float32)
    qamat = np.zeros((len(df),6), dtype=np.float16)
    nattmpt = np.zeros((len(df),5), dtype=np.float16)
    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})
    
    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp', 'content_type_id', 'user_answer']
    if not update:
        itercols = [c for c in itercols if c != 'answered_correctly']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp, ctype, ua = row
        else:
            u, pexp, eltim, cid, tag, tcid, tstmp, ctype, ua = row
            
        try:
            ukey = kdicts['userKey'][u]
        except:
            ukey = kdicts['userKey'][u] = kdicts['userCtr']
            kdicts['userCtr'] += 1
            
        try:
            ckey = kdicts['usercontentKey'][u][cid]
        except:
            ckey = kdicts['usercontentKey'][u][cid] = kdicts['usercontentCtr']
            kdicts['usercontentCtr'] += 1
            
        if ctype == 0:
            bid = pdicts['bdict'][cid]
            newbid = bid == pdicts['user_id'].item(ukey, 22)
        tstmp = int(round(tstmp/1000))
        
        if update:
            if ctype==1:
                lectcid, lpart, ltype_of = cid, ldict['part'][cid], ldict['type_of'][cid]
                pdicts['lect_time_mat'][ukey, 0] += 1 # pdicts['cum_lecture_ct'][u] += 1
                pdicts['lect_time_mat'][ukey, 1] = lpart # pdicts['prev_lecture_part'][u] = lpart
                pdicts['lect_time_mat'][ukey, 4] = pdicts['lect_time_mat'][ukey, 3] # pdicts['prev_lecture_timestamp3'][u] = pdicts['prev_lecture_timestamp2'][u]
                pdicts['lect_time_mat'][ukey, 3] = pdicts['lect_time_mat'][ukey, 2] # pdicts['prev_lecture_timestamp2'][u] = pdicts['prev_lecture_timestamp'][u]
                pdicts['lect_time_mat'][ukey, 2] = tstmp # pdicts['prev_lecture_timestamp'][u] = tstmp 
                pdicts['lect_time_mat'][ukey, 5] = 1 # pdicts['cum_ques_since_lecture'][u] = 1
                continue
            else:
                if pdicts['lect_time_mat'][ukey, 5] == 1: #if pdicts['cum_ques_since_lecture'][u] == 1:
                    time_since = int(round(tstmp - pdicts['lect_time_mat'][ukey, 2]))/1000 #    time_since = int(round(tstmp - pdicts['prev_lecture_timestamp'][u]))/1000
                    pdicts['lect_time_mat'][ukey, 7] += time_since#    pdicts['cum_lecture_time'][u] += time_since
                    pdicts['lect_time_mat'][ukey, 6] = time_since #    pdicts['prev_lecture_time'][u] = time_since
                if pdicts['lect_time_mat'][ukey, 5]>0: #if pdicts['cum_ques_since_lecture'][u]>0:
                    pdicts['lect_time_mat'][ukey, 5] += 1#    pdicts['cum_ques_since_lecture'][u] += 1
        
        '''
        container_curr:0                            container_reattempt_count:2
        container_reattempt_count_lag1:3            container_unq_ques_lag1:4
        container_lag0:5    container_lag1:6        container_lag2:7
        '''
            
        if (tcid + 1 == pdicts['container'][ukey, 0]) or (cid in pdicts['container_unq_ques'][u]):
            if cid in pdicts['container_unq_ques'][u]:
                pdicts['container'][ukey, 0] = tcid + 1
            pdicts['container'][ukey, 2] += 1
            pdicts['container_unq_ques'][u].add(cid)
        else:
            pdicts['container'][ukey, 3] = pdicts['container'][ukey, 2]
            pdicts['container'][ukey, 4] = len(pdicts['container_unq_ques'][u])
            pdicts['container'][ukey, 0] = tcid + 1
            pdicts['container'][ukey, 2] = 1
            pdicts['container_unq_ques'][u] = set([cid])
            pdicts['container'][ukey, 7] = pdicts['container'][ukey, 6]
            pdicts['container'][ukey, 6] = pdicts['container'][ukey, 5]
            pdicts['container'][ukey, 5] = tstmp 
                    
        # 4 -> 6 -> 3 -> 2 -> 0 -> 1
        lectct[cnt] = pdicts['lect_time_mat'].item(ukey, 0), pdicts['lect_time_mat'].item(ukey, 1), \
                    pdicts['lect_time_mat'].item(ukey, 6), pdicts['lect_time_mat'].item(ukey, 5), \
                    tstmp - pdicts['lect_time_mat'].item(ukey, 2), \
                    tstmp - pdicts['lect_time_mat'].item(ukey, 3), \
                    tstmp - pdicts['lect_time_mat'].item(ukey, 4)
        lectavg[cnt] = pdicts['lect_time_mat'].item(ukey, 7) / (pdicts['lect_time_mat'].item(ukey, 0) + 0.01), \
                        pdicts['lect_time_mat'].item(ukey, 0) / ( pdicts['user_id'].item(ukey, 1) +0.01), \
                        pdicts['lect_time_mat'].item(ukey, 5) / ( pdicts['user_id'].item(ukey, 1) +0.01)
        
        acsu[cnt] = pdicts['user_id'].item(ukey, 0)  # pdicts['answered_correctly_sum_u_dict'][u]
        cu[cnt] = pdicts['user_id'].item(ukey, 1)   # pdicts['count_u_dict'][u]
        if pdicts['user_id'].item(ukey, 1) > 0:
            decay_ratio = pdicts['decay'].item(ukey, 0) / (pdicts['decay'].item(ukey, 1) + 0.001)
            acsudec[cnt] = decay_ratio, \
                            decay_ratio - pdicts['user_id'].item(ukey, 0)/ (pdicts['user_id'].item(ukey, 1) + 0.001)
        
        nattmpt[cnt] = pdicts['container'][ukey, 2] / len(pdicts['container_unq_ques'][u]), \
                            pdicts['container'][ukey, 3] / (pdicts['container'][ukey, 4] + 0.01), \
                            tstmp  - pdicts['container'][ukey, 7], \
                            tstmp  - pdicts['container'][ukey, 6], \
                            tstmp  - pdicts['container'][ukey, 5]
                            
        expacsu[cnt] = pdicts['user_id'].item(ukey, 2) # pdicts['pexp_answered_correctly_sum_u_dict'][u]
        expcu[cnt] = pdicts['user_id'].item(ukey, 3) # pdicts['pexp_count_u_dict'][u]
        pcu[cnt] =  pdicts['content_id'].item(ckey, 0) # pdicts['content_id_answered_correctly_prev'][u][cid] 
        cidacsu[cnt] = pdicts['content_id'].item(ckey, 1) # pdicts['content_id_answered_correctly_sum_u_dict'][u][cid]
        cidcu[cnt] = pdicts['content_id'].item(ckey, 2) # pdicts['content_id_count_u_dict'][u][cid]
        partsdict[part]['acsu'][cnt]  =  pdicts['user_id'].item(ukey, 4 + part*2)# pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        partsdict[part]['cu'][cnt] = pdicts['user_id'].item(ukey, 4 + part*2 + 1) # pdicts[f'{part}p_count_u_dict'][u]
        for p in range(1,8):
            if p == part: continue
            partsdict[p]['cu'][cnt] = pdicts['user_id'].item(ukey, 4 + p*2 + 1)
            partsdict[p]['acsu'][cnt]  = pdicts['user_id'].item(ukey, 4 + p*2)
        acsb[cnt] = pdicts['user_id'].item(ukey, 20)  # pdicts['answered_correctly_sum_u_dict'][u]
        cb[cnt] = pdicts['user_id'].item(ukey, 21)   # pdicts['count_u_dict'][u]        
        tstamp[cnt] = [tstmp - pdicts['lag_time'].item(ukey, i) for i in [0,1,2,4,9]]
        tstavg[cnt] = pdicts['lag_time_cum'].item(ukey, 0)/ (pdicts['user_id'][ukey, 1]+0.1)
        
        '''
        qamat[cnt] = pdicts['qaRankcum'][u] / (pdicts['count_u_dict'][u] + 0.01), \
                pdicts['qaRatiocum'][u] / (pdicts['count_u_dict'][u] + 0.01), \
                pdicts['qaRankcum'][u] / (pdicts['qaRankCorrectcum'][u] + 0.01), \
                pdicts['qaRatiocum'][u] / (pdicts['qaRatioCorrectcum'][u] + 0.01), \
                pdicts['qaRankFirstcum'][u] / (pdicts['qaRankCorrectFirstcum'][u] + 0.01), \
                pdicts['qaRatioFirstcum'][u] / (pdicts['qaRatioCorrectFirstcum'][u] + 0.01)
        '''
        qamat[cnt] = pdicts['rank'].item(ukey, 1) / (pdicts['user_id'].item(ukey, 1) + 0.01), \
                pdicts['rank'].item(ukey, 3) / (pdicts['user_id'].item(ukey, 1) + 0.01), \
                pdicts['rank'].item(ukey, 1) / (pdicts['rank'].item(ukey, 0) + 0.01), \
                pdicts['rank'].item(ukey, 3) / (pdicts['rank'].item(ukey, 2) + 0.01), \
                pdicts['rank'].item(ukey, 5) / (pdicts['rank'].item(ukey, 4) + 0.01), \
                pdicts['rank'].item(ukey, 7) / (pdicts['rank'].item(ukey, 6) + 0.01)
        
        if update:
            try:
                pdicts['rank'][ukey, 0] += pdicts['qaRank'][(cid, pdicts['qaCorrect'][cid])]
                pdicts['rank'][ukey, 2] += pdicts['qaRatio'][(cid, pdicts['qaCorrect'][cid])]
            except:
                pdicts['rank'][ukey, 0] += 0
                pdicts['rank'][ukey, 2] += 1.
            try:
                pdicts['rank'][ukey, 1] += pdicts['qaRank'][(cid, ua)]
                pdicts['rank'][ukey, 3] += pdicts['qaRatio'][(cid, ua)]
            except:
                pdicts['rank'][ukey, 1] += 4.
                pdicts['rank'][ukey, 3] += 0.1
            
            # Only add this on first attempts, and track first attempt success rate
            if pdicts['content_id'].item(ckey, 1) == 0:
                try:
                    pdicts['rank'][ukey, 4] += pdicts['qaRankFirst'][(cid, pdicts['qaCorrect'][cid])]
                    pdicts['rank'][ukey, 6] += pdicts['qaRatioFirst'][(cid, pdicts['qaCorrect'][cid])]
                except:
                    pdicts['rank'][ukey, 4] += 0
                    pdicts['rank'][ukey, 6] += 1.
                try:
                    pdicts['rank'][ukey, 5] += pdicts['qaRankFirst'][(cid, ua)]
                    pdicts['rank'][ukey, 7] += pdicts['qaRatioFirst'][(cid, ua)]
                except:
                    pdicts['rank'][ukey, 5] += 4.
                    pdicts['rank'][ukey, 7] += 0.1
            
            ### New
            lag0daysdecay = (max(30, (tstmp - pdicts['lag_time'][ukey, 0])/(3600*24))) / 60
            pdicts['decay'][ukey, 1] = 1 + (1-lag0daysdecay) * pdicts['decay'].item(ukey, 1)
            pdicts['decay'][ukey, 0] = yprev + (1-lag0daysdecay) * pdicts['decay'].item(ukey, 0)
            
            for i in list(range(1, 10))[::-1]:
                pdicts['lag_time'][ukey, i] = pdicts['lag_time'][ukey, i-1] 
            pdicts['lag_time'][ukey, 0] = tstmp
            pdicts['lag_time_cum'][ukey, 0] += tstmp
            
            
            pdicts['user_id'][ukey, 1] += 1
            pdicts['user_id'][ukey, 4 + part*2 + 1] += 1
            pdicts['content_id'][ckey, 2] += 1
            pdicts['content_id'][ckey, 0] = yprev + 1
            
            pdicts['user_id'][ukey, 21] = 1 if newbid else pdicts['user_id'][ukey, 21] + 1
            if newbid : pdicts['user_id'][ukey, 20] = 0
            
            if yprev: 
                pdicts['user_id'][ukey, 0] += 1
                pdicts['content_id'][ckey, 1] += 1
                pdicts['user_id'][ukey, 4 + part*2] += 1
                pdicts['user_id'][ukey, 20] += 1
            if pexp:
                if yprev: 
                    pdicts['user_id'][ukey, 2] += yprev 
                pdicts['user_id'][ukey, 3] += 1
            pdicts['user_id'][ukey, 22] = bid
                
    for t1, (matcu, matascu) in enumerate(zip([cu, expcu, cidcu, cb], 
                                              [acsu, expacsu, cidacsu, acsb])):
        df[f'counts___feat{t1}'] = matcu
        df[f'avgcorrect___feat{t1}'] =  (matascu / (matcu + 0.001)).astype(np.float16)
        #gc.collect()
    df['cid_answered_correctly'] = acsu
    df[[f'lag_content_time{i}' for i in [0,1,2,5,10]]] = tstamp.astype(np.uint32)
    df['lag_content_avgtime'] = tstavg
    df[[f'lecture_stats_{i}' for i in range(7)]] = lectct
    df[[f'lecture_stats_{i}' for i in range(7)]] = df[[f'lecture_stats_{i}' for i in range(7)]].astype(np.int32)
    df[[f'lecture_stats_{i}' for i in range(7,10)]] = lectavg.astype(np.float32) 
    df[[f'lecture_stats_{i}' for i in range(7,10)]] = df[[f'lecture_stats_{i}' for i in range(7,10)]].astype(np.float32)
    df[[f'rank_stats_{i}' for i in range(6)]] = qamat
    df[[f'rank_stats_{i}' for i in range(6)]] = df[[f'rank_stats_{i}' for i in range(6)]].astype(np.float32)
    df[[f'decayed_avg_correct{i}' for i in range(2)]] = acsudec
    df[[f'nattempt_lag_{i}' for i in range(5)]] = nattmpt
    #df[['ctunique_sum', 'ctunique_attempt_ration']] = ctunq
    del cu, expcu, acsu, expacsu
    for t, i in enumerate(range(1,8)):  
        df[f'counts___feat{t1+t+1}'] = partsdict[i]['cu']
        df[f'avgcorrect___feat{t1+t+1}'] =  (partsdict[i]['acsu']  / (partsdict[i]['cu'] + 0.001)).astype(np.float16)
        del partsdict[i]
        #gc.collect()
    df['content_id_answered_correctly_prev'] = pcu
    
    return df

def add_user_feats_without_update(df, pdicts, kdicts, update=False):
    df = add_user_feats(df, pdicts, kdicts, update)
    return df

def update_user_feats(df, pdicts, kdicts):
    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp', 'content_type_id']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for row in df[itercols].fillna(False).values:
        u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp, ctype = row
        if ctype == 0:
            try:
                ukey = kdicts['userKey'][u]
            except:
                ukey = kdicts['userKey'][u] = kdicts['userCtr']
                kdicts['userCtr'] += 1
                
            try:
                ckey = kdicts['usercontentKey'][u][cid]
            except:
                ckey = kdicts['usercontentKey'][u][cid] = kdicts['usercontentCtr']
                kdicts['usercontentCtr'] += 1
                
            bid = pdicts['bdict'][cid]
            newbid = bid == pdicts['user_id'].item(ukey, 22)
            tstmp = int(round(tstmp/1000))
            
            for i in list(range(1, 10))[::-1]:
                pdicts['lag_time'][ukey, i] = pdicts['lag_time'][ukey, i-1] 
            pdicts['lag_time'][ukey, 0] = tstmp
            pdicts['lag_time_cum'][ukey, 0] += tstmp
            pdicts['user_id'][ukey, 1] += 1
            pdicts['user_id'][ukey, 4 + part*2 + 1] += 1
            pdicts['content_id'][ckey, 2] += 1
            pdicts['content_id'][ckey, 0] = yprev + 1
            
            pdicts['user_id'][ukey, 21] = 1 if newbid else pdicts['user_id'][ukey, 21] + 1
            if newbid : pdicts['user_id'][ukey, 20] = 0
            
            if yprev: 
                pdicts['user_id'][ukey, 0] += 1
                pdicts['content_id'][ckey, 1] += 1
                pdicts['user_id'][ukey, 4 + part*2] += 1
                pdicts['user_id'][ukey, 20] += 1
            if pexp:
                if yprev: 
                    pdicts['user_id'][ukey, 2] += yprev 
                pdicts['user_id'][ukey, 3] += 1
            pdicts['user_id'][ukey, 22] = bid
            

CUT=0
DIR='valfull'
VERSION='V17'
debug = False
validaten_flg = False
FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', 'timestamp', 'user_answer']
valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]

if debug:
    train = train[:100000]
    valid = train[:10000]
    
    
if False:
    dumpobj(f'data/{DIR}/cv{CUT+1}_valid.pk', valid)
    dumpobj(f'data/{DIR}/cv{CUT+1}_train.pk', train)
    v= pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')
    v.to_feather()
    dumpobj(f'data/valfull/valid_{DIR}_{VERSION}.sav', v) 

questions_df = pd.read_csv('data/questions.csv')
ldf = pd.read_csv('data/lectures.csv')
ldf.type_of = ldf.type_of.str.replace(' ', '_')
ldict = ldf.set_index('lecture_id').to_dict()
lecture_types = [t for t in ldf.type_of.unique() if t!= 'starter']

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

# Merge questions
keepcols = ['question_id', 'part', 'tags', 'bundle_id', 'correct_answer']
train = pd.merge(train, questions_df[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[keepcols], left_on = 'content_id', right_on = 'question_id', how = 'left')
formatcols =  ['question_id', 'part', 'bundle_id', 'correct_answer', 'user_answer']
train[formatcols] = train[formatcols].fillna(0).astype(np.int16)
valid[formatcols] = valid[formatcols].fillna(0).astype(np.int16)
train.tags = train.tags.fillna('')
valid.tags = valid.tags.fillna('')

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

# answered correctly average for each content
content_df1 = train.query('content_type_id == 0')[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean', 'count']).astype(np.float16).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c', 'answered_correctly_ct_c']
content_df2 = train.query('content_type_id == 0').groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float16).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df3 = train.query('content_type_id == 0')[['content_id', 'user_id','answered_correctly']].drop_duplicates(keep='first')
content_df3 = content_df3[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df3.columns = ['content_id', 'answered_correctly_first_avg_c']
content_df4 = train.query('content_type_id == 0')[['content_id', 'user_id','answered_correctly']].drop_duplicates(keep='last')
content_df4 = content_df4[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df4.columns = ['content_id', 'answered_correctly_last_avg_c']
content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df  = pd.merge(content_df, content_df3, on = 'content_id')
content_df  = pd.merge(content_df, content_df4, on = 'content_id')
content_df.columns
del content_df1, content_df2, content_df3, content_df4
gc.collect()

train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")



kdicts = getKeys(train)
pdicts = {
          'content_id' : np.zeros( (int(kdicts['usercontentCtr'] *1.2) , 3), dtype= np.uint8),
          'bdict' : questions_df.set_index('question_id')['bundle_id'].to_dict(),
          'user_id' : np.zeros((int(kdicts['userCtr'] *1.2) , 40), dtype= np.int16),
          'lag_time': np.zeros((int(kdicts['userCtr'] *1.2) , 10), dtype= np.uint32),
          'lag_time_cum': np.zeros((int(kdicts['userCtr'] *1.2) , 1), dtype= np.uint64),
          'lect_time_mat': np.zeros((int(kdicts['userCtr'] *1.2) , 8), dtype= np.uint32),
          'lect_time_cum': np.zeros((int(kdicts['userCtr'] *1.2) , 1), dtype= np.uint64),
          'rank': np.zeros((int(kdicts['userCtr'] *1.2) , 8), dtype= np.float16),
          'decay': np.zeros((int(kdicts['userCtr'] *1.2) , 2), dtype= np.float16),
          'container': np.zeros((int(kdicts['userCtr'] *1.2) , 8), dtype= np.float16),
          'container_unq_ques' : defaultdict(set),
          'qaRank' : qaRank,
          'qaRatio' : qaRatio,
          'qaRankFirst' : qaRankFirst, 
          'qaRatioFirst' : qaRatioFirst, 
          'qaCorrect': qaCorrect,
          }

train = add_user_feats(train, pdicts, kdicts)
valid = add_user_feats(valid, pdicts, kdicts)

valid[[f'decayed_avg_correct{i}' for i in range(2)]]

train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)
#train = train.loc[train.content_type_id == False].reset_index(drop=True)
#valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

def split_tags(s, dummy = 188):
    try:
        s = s.split(' ')
        l = list(map(int, s))
        l += [dummy]*(6-len(l))
        return l
    except:
        return [dummy]*6

valid[[f'tag{i}' for i in range(6)]] = np.array(valid.tags.apply(split_tags).tolist()).astype(np.uint16)
train[[f'tag{i}' for i in range(6)]] = np.array(train.tags.apply(split_tags).tolist()).astype(np.uint16)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

train[[f'rank_stats_diff_{i}' for i in [0,1]]] = (train[[f'rank_stats_{i}' for i in [0,3]]].values - \
                                            train[[f'rank_stats_{i}' for i in [2,1]]].values).astype(np.float32)

valid[[f'rank_stats_diff_{i}' for i in [0,1]]] = (valid[[f'rank_stats_{i}' for i in [0,3]]].values - \
                                            valid[[f'rank_stats_{i}' for i in [2,1]]].values).astype(np.float32)

TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', 'cid_answered_correctly', \
         'content_id', 'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', 'correct_answer', \
         'answered_correctly_ct_c', 'answered_correctly_last_avg_c', 'lag_content_avgtime']
FEATS += [f'counts___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
FEATS += [f'lag_content_time{i}' for i in [0,1,2,5,10]]
FEATS += [f'tag{i}' for i in range(6)]
#FEATS += ['ctunique_sum', 'ctunique_attempt_ration']
FEATS += [f'lecture_stats_{i}' for i in range(10)]
FEATS += [f'rank_stats_{i}' for i in range(6)]
FEATS += [f'rank_stats_diff_{i}' for i in range(2)]
FEATS += [f'nattempt_lag_{i}' for i in range(5)]
FEATS += [f'decayed_avg_correct{i}' for i in range(2)]

y_tr = train[TARGET]
y_va = valid[TARGET]
_=gc.collect()
X_train_np = train[FEATS].values.astype(np.float32)
X_valid_np = valid[FEATS].values.astype(np.float32)
_=gc.collect()

categoricals = ['part', 'content_id'] + [f'tag{i}' for i in range(6)]
lgb_train = lgb.Dataset(X_train_np, y_tr, categorical_feature = categoricals, feature_name=FEATS)
del X_train_np, y_tr
_=gc.collect()
lgb_valid = lgb.Dataset(X_valid_np, y_va, categorical_feature = categoricals, feature_name=FEATS)
_=gc.collect()

if True:
    joblib.dump(valid, f'data/valfull/cv1_{VERSION}_valid.pk')
    del valid, train
    gc.collect()
    dumpobj(f'data/valfull/cv1_{VERSION}_prior_question_elapsed_time_mean.pk', prior_question_elapsed_time_mean)
    dumpobj(f'data/valfull/cv1_{VERSION}_content_df.pk', content_df)
    dumpobj(f'data/valfull/cv1_{VERSION}_FEATS.pk', FEATS)
    dumpobj(f'data/valfull/cv1_{VERSION}_TARGET.pk', TARGET)
    dumpobj(f'data/valfull/cv1_{VERSION}_cut0_val.pk', pdicts)  
    del pdicts
    gc.collect()
    usercontentKeyMat = np.zeros((kdicts['usercontentCtr'], 2), np.uint32)
    for k1, v1 in tqdm(kdicts['usercontentKey'].items(), 
                       total = len(kdicts['usercontentKey'].keys())):
        for k2, v2 in v1.items():
            usercontentKeyMat[v2] = k1,k2
    dumpobj(f'data/valfull/cv1_{VERSION}_usercontentKeyMat.pk', usercontentKeyMat)
    del kdicts['usercontentKey']#
    del usercontentKeyMat#
    gc.collect()
    for k, v in kdicts.items():
        print(f'Object {k} size {sys.getsizeof(v)}')
        try:
            joblib.dump(v, f'data/valfull/kdicts_{k}_{VERSION}_cut0_val.sav')  
        except:
            v = dict(v.copy())
            joblib.dump(v, f'data/valfull/kdicts_{k}_{VERSION}_cut0_val.sav')  
            del v
            gc.collect()
    del kdicts
    gc.collect()

model = lgb.train(
                    {'objective': 'binary', 
                     'min_data_per_group':5000,
                     'min_data_in_leaf' : 300,
                     'learning_rate': 0.1},
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=50,
                    num_boost_round=8000,
                    early_stopping_rounds=100
                )
# print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)
model.save_model(f'data/valfull/model_{VERSION}_valfull_cut0_val.pk')

valid = joblib.load(f'data/valfull/cv1_{VERSION}_valid.pk')
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))

#### New 
# Some container lag feature
# Cumulative prev time vs global prev time
# Rolling ratio of correct answers vs average correct
# decay avg correct

'''
[100]   training's binary_logloss: 0.525173     valid_1's binary_logloss: 0.529532
[500]	training's binary_logloss: 0.517292	valid_1's binary_logloss: 0.52235
[1000]	training's binary_logloss: 0.513778	valid_1's binary_logloss: 0.519638
[1500]	training's binary_logloss: 0.511745	valid_1's binary_logloss: 0.518352
[2000]	training's binary_logloss: 0.510237	valid_1's binary_logloss: 0.517575
[2500]	training's binary_logloss: 0.509053	valid_1's binary_logloss: 0.517001
[3000]	training's binary_logloss: 0.508217	valid_1's binary_logloss: 0.516666
[3500]	training's binary_logloss: 0.507492	valid_1's binary_logloss: 0.516421
[4000]	training's binary_logloss: 0.506834	valid_1's binary_logloss: 0.516215
[4500]	training's binary_logloss: 0.506178	valid_1's binary_logloss: 0.516013
[5000]	training's binary_logloss: 0.505617	valid_1's binary_logloss: 0.515858
[5500]	training's binary_logloss: 0.505065	valid_1's binary_logloss: 0.515741
[6000]	training's binary_logloss: 0.504587	valid_1's binary_logloss: 0.515646
[6500]	training's binary_logloss: 0.504138	valid_1's binary_logloss: 0.51558
[7000]	training's binary_logloss: 0.503697	valid_1's binary_logloss: 0.515502
[7450]	training's binary_logloss: 0.503304	valid_1's binary_logloss: 0.515445
Early stopping, best iteration is:
[7410]	training's binary_logloss: 0.503332	valid_1's binary_logloss: 0.515445
auc: 0.7970130719101044
'''

'''
Training until validation scores don't improve for 10 rounds
[100]	training's binary_logloss: 0.53044	valid_1's binary_logloss: 0.535057
[500]	training's binary_logloss: 0.525056	valid_1's binary_logloss: 0.529787
[550]	training's binary_logloss: 0.524822	valid_1's binary_logloss: 0.529574
[1000]	training's binary_logloss: 0.523139	valid_1's binary_logloss: 0.528041
[1500]	training's binary_logloss: 0.52194	valid_1's binary_logloss: 0.527009
[2000]	training's binary_logloss: 0.521004	valid_1's binary_logloss: 0.526214
[3000]	training's binary_logloss: 0.519718	valid_1's binary_logloss: 0.525267
Early stopping, best iteration is:
[3642]	training's binary_logloss: 0.518991	valid_1's binary_logloss: 0.524735
auc: 0.7875689487502573
'''


'''
Val
[100]	training's binary_logloss: 0.52264	valid_1's binary_logloss: 0.537271
[200]	training's binary_logloss: 0.518315	valid_1's binary_logloss: 0.534111
[400]	training's binary_logloss: 0.514488	valid_1's binary_logloss: 0.531984
[600]	training's binary_logloss: 0.512098	valid_1's binary_logloss: 0.531045
[800]	training's binary_logloss: 0.510297	valid_1's binary_logloss: 0.530444
[1000]	training's binary_logloss: 0.50887	valid_1's binary_logloss: 0.530095
[1200]	training's binary_logloss: 0.507698	valid_1's binary_logloss: 0.52987
[1400]	training's binary_logloss: 0.506359	valid_1's binary_logloss: 0.529599
[1600]	training's binary_logloss: 0.505324	valid_1's binary_logloss: 0.529447
[1900]	training's binary_logloss: 0.503659	valid_1's binary_logloss: 0.529187
Early stopping, best iteration is:
[1902]	training's binary_logloss: 0.503648	valid_1's binary_logloss: 0.529184
auc: 0.7882457187238955
'''

'''
Val
[100]	training's binary_logloss: 0.525884	valid_1's binary_logloss: 0.540276
[300]	training's binary_logloss: 0.519514	valid_1's binary_logloss: 0.535896
[500]	training's binary_logloss: 0.516681	valid_1's binary_logloss: 0.534593
[700]	training's binary_logloss: 0.514866	valid_1's binary_logloss: 0.533965
[900]	training's binary_logloss: 0.51344	valid_1's binary_logloss: 0.533586
[1100]	training's binary_logloss: 0.512103	valid_1's binary_logloss: 0.533326
[1200]	training's binary_logloss: 0.51146	valid_1's binary_logloss: 0.533154
[1250]	training's binary_logloss: 0.511243	valid_1's binary_logloss: 0.533141
Early stopping, best iteration is:
[1234]	training's binary_logloss: 0.511298	valid_1's binary_logloss: 0.533135
auc: 0.7839775905164138
'''
'''
VAL
Training until validation scores don't improve for 10 rounds
[50]	training's binary_logloss: 0.532701	valid_1's binary_logloss: 0.546426
[100]	training's binary_logloss: 0.528141	valid_1's binary_logloss: 0.541936
[150]	training's binary_logloss: 0.526248	valid_1's binary_logloss: 0.540295
[200]	training's binary_logloss: 0.52513	valid_1's binary_logloss: 0.539406
[250]	training's binary_logloss: 0.524355	valid_1's binary_logloss: 0.538786
[300]	training's binary_logloss: 0.523762	valid_1's binary_logloss: 0.538398
[350]	training's binary_logloss: 0.523182	valid_1's binary_logloss: 0.538009
[400]	training's binary_logloss: 0.52273	valid_1's binary_logloss: 0.537773
[450]	training's binary_logloss: 0.522271	valid_1's binary_logloss: 0.53756
[500]	training's binary_logloss: 0.521877	valid_1's binary_logloss: 0.537338
[550]	training's binary_logloss: 0.521524	valid_1's binary_logloss: 0.537159
[600]	training's binary_logloss: 0.521207	valid_1's binary_logloss: 0.537066
[650]	training's binary_logloss: 0.520874	valid_1's binary_logloss: 0.536936
Early stopping, best iteration is:
[672]	training's binary_logloss: 0.52071	valid_1's binary_logloss: 0.53686
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
auc: 0.7800607676363841
'''



