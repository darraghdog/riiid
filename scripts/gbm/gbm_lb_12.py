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
from scripts.utils import Iter_Valid, dumpobj, loadobj
from sklearn.externals import joblib
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# user stats features with loops
def getKeys(train):
    kd = {#'bundlekey' : questions_df.set_index('question_id')['bundle_id'].to_dict(), 
                'userCtr': 1, 
                'usercontentCtr': 1,
                #'userbundleCtr': 1, 
                'userCtr': 1,
                'usercontentKey' : defaultdict(lambda: {}),
                'userKey' : {}}#,
                #'userbundleKey' : defaultdict(lambda: {})}
    for row in tqdm(train[['user_id', 'content_id']].values):
        user, cont = row 
        #bund = kd['bundlekey'][cont]
        if cont not in kd['usercontentKey'][user]:
            kd['usercontentKey'][user][cont] = kd['usercontentCtr']
            kd['usercontentCtr'] += 1
        #if bund not in kd['userbundleKey'][user]:
        #    kd['userbundleKey'][user][bund] = kd['userbundleCtr']
        #    kd['userbundleCtr'] += 1
        if user not in kd['userKey']:
            kd['userKey'][user] = kd['userCtr']
            kd['userCtr'] += 1
    return kd


# funcs for user stats with loop
def add_user_feats(df, pdicts, kdicts, update = True):
    
    acsu = np.zeros(len(df), dtype=np.uint32)
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
    tstamp = np.zeros(len(df), dtype=np.float32)
    tstavg = np.zeros(len(df), dtype=np.float32)
    
    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})
    
    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp']
    if not update:
        itercols = [c for c in itercols if c != 'answered_correctly']
    
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp = row
        else:
            u, pexp, eltim, cid, tag, tcid, tstmp = row
        
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
        # tags = tag.split()
                    
        acsu[cnt] = pdicts['user_id'].item(ukey, 0)  # pdicts['answered_correctly_sum_u_dict'][u]
        cu[cnt] = pdicts['user_id'].item(ukey, 1)   # pdicts['count_u_dict'][u]
        expacsu[cnt] = pdicts['user_id'].item(ukey, 2) # pdicts['pexp_answered_correctly_sum_u_dict'][u]
        expcu[cnt] = pdicts['user_id'].item(ukey, 3) # pdicts['pexp_count_u_dict'][u]
        pcu[cnt] =  pdicts['content_id'].item(ckey, 0) # pdicts['content_id_answered_correctly_prev'][u][cid] 
        cidacsu[cnt] = pdicts['content_id'].item(ckey, 1) # pdicts['content_id_answered_correctly_sum_u_dict'][u][cid]
        cidcu[cnt] = pdicts['content_id'].item(ckey, 2) # pdicts['content_id_count_u_dict'][u][cid]
        #bidacsu[cnt] = pdicts['bundle_id'].item(bkey, 0) # pdicts['bundle_id_answered_correctly_sum_u_dict'][u][bid]
        #bidcu[cnt] = pdicts['bundle_id'].item(bkey, 1) # pdicts['bundle_id_count_u_dict'][u][bid]
        partsdict[part]['acsu'][cnt]  =  pdicts['user_id'].item(ukey, 4 + part*2)# pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        partsdict[part]['cu'][cnt] = pdicts['user_id'].item(ukey, 4 + part*2 + 1) # pdicts[f'{part}p_count_u_dict'][u]
        acsb[cnt] = pdicts['user_id'].item(ukey, 20)  # pdicts['answered_correctly_sum_u_dict'][u]
        cb[cnt] = pdicts['user_id'].item(ukey, 21)   # pdicts['count_u_dict'][u]
        tstamp[cnt] = tstmp - pdicts['lag_time'].item(ukey, 0)  
        tstavg[cnt] = pdicts['lag_time'][ukey, 1] / (pdicts['user_id'].item(ukey, 1) + 0.1)
        
        if update:
            pdicts['lag_time'][ukey, 0] = tstmp
            pdicts['lag_time'][ukey, 1] += tstmp
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
    df['lag_content_time'] = tstamp
    df['lag_content_avgtime'] = tstavg
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
    filtcols = ['user_id','answered_correctly', 'part', 'prior_question_had_explanation', 'content_id', 'content_type_id']

    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for row in df[filtcols].fillna(False).values:
        u, yprev, part, pexp, cid, ctype = row
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
            
            pdicts['lag_time'][ukey, 0] = tstmp
            pdicts['lag_time'][ukey, 1] += tstmp
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
DIR='valfull' # 'valfull''
VERSION='V12'
debug = False
validaten_flg = False
FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', \
                   'prior_question_had_explanation', 'task_container_id', 'timestamp']
valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]

if False:
    dumpobj(f'data/{DIR}/cv{CUT+1}_valid.pk', valid)
    dumpobj(f'data/{DIR}/cv{CUT+1}_train.pk', train)
    v= pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')
    v.to_feather()
    dumpobj(f'data/valfull/valid_{DIR}_{VERSION}.sav', v) 

questions_df = pd.read_csv('data/questions.csv')
if debug:
    train = train[:1000000]
    valid = valid[:10000]
    
train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

# answered correctly average for each content
content_df1 = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean', 'count']).astype(np.float16).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c', 'answered_correctly_ct_c']
content_df2 = train.groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float16).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df3 = train[['content_id', 'user_id','answered_correctly']].drop_duplicates(keep='first')
content_df3 = content_df3[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df3.columns = ['content_id', 'answered_correctly_first_avg_c']
content_df4 = train[['content_id', 'user_id','answered_correctly']].drop_duplicates(keep='last')
content_df4 = content_df4[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df4.columns = ['content_id', 'answered_correctly_last_avg_c']
content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df  = pd.merge(content_df, content_df3, on = 'content_id')
content_df  = pd.merge(content_df, content_df4, on = 'content_id')
content_df.columns

train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

# part
questions_df['part'] = questions_df['part'].astype(np.uint8)
train = pd.merge(train, questions_df[['question_id', 'part', 'tags']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part', 'tags']], left_on = 'content_id', right_on = 'question_id', how = 'left')




kdicts = getKeys(train)
pdicts = {
          'content_id' : np.zeros( (int(kdicts['usercontentCtr'] *1.2) , 3), dtype= np.uint8),
          'bdict' : questions_df.set_index('question_id')['bundle_id'].to_dict(),
          'user_id' : np.zeros((int(kdicts['userCtr'] *1.2) , 40), dtype= np.int16),
          'lag_time': np.zeros((int(kdicts['userCtr'] *1.2) , 4), dtype= np.float32),
          }

train = add_user_feats(train, pdicts, kdicts)
valid = add_user_feats(valid, pdicts, kdicts)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

TARGET = 'answered_correctly'

FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', 'cid_answered_correctly', \
         'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', 'lag_content_time', \
         'answered_correctly_ct_c', 'answered_correctly_last_avg_c', 'lag_content_avgtime']
FEATS += [f'counts___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]

    #dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
#train.drop(dro_cols, axis=1, inplace=True)
#valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()

lgb_train = lgb.Dataset(train[FEATS], y_tr)
lgb_valid = lgb.Dataset(valid[FEATS], y_va)
_=gc.collect()


model = lgb.train(
                    {'objective': 'binary', 'learning_rate': 0.1, 'metric': 'auc'}, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    early_stopping_rounds=10
                )
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)



if True:
    model.save_model(f'data/valfull/model_{VERSION}_valfull_cut0_val.pk')
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
    '''
    userbundleKeyMat = np.zeros((kdicts['userbundleCtr'], 2), np.uint64)
    for k1, v1 in tqdm(kdicts['userbundleKey'].items(), 
                       total = len(kdicts['userbundleKey'].keys())):
        for k2, v2 in v1.items():
            userbundleKeyMat[v2] = k1,k2
    '''
    usercontentKeyMat = np.zeros((kdicts['usercontentCtr'], 2), np.uint32)
    for k1, v1 in tqdm(kdicts['usercontentKey'].items(), 
                       total = len(kdicts['usercontentKey'].keys())):
        for k2, v2 in v1.items():
            usercontentKeyMat[v2] = k1,k2
    dumpobj(f'data/valfull/cv1_{VERSION}_usercontentKeyMat.pk', usercontentKeyMat)
    #dumpobj(f'data/valfull/cv1_{VERSION}_userbundleKeyMat.pk', userbundleKeyMat)
    del kdicts['usercontentKey']#, kdicts['userbundleKey']
    del usercontentKeyMat#, userbundleKeyMat
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



