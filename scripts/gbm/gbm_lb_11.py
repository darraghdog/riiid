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
    expacsu = np.zeros(len(df), dtype=np.uint32)
    expcu = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    #bidacsu = np.zeros(len(df), dtype=np.uint32)
    #bidcu = np.zeros(len(df), dtype=np.uint32)
    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})
    
    if update:
        itercols = ['user_id','answered_correctly', 'part', 'prior_question_had_explanation', 'content_id']
    else:
        itercols = ['user_id', 'part', 'prior_question_had_explanation', 'content_id']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, cid = row
        else:
            u, part, pexp, cid = row

        #bid = kdicts['bundlekey'][cid]
        
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
            
        #try:
        #    bkey = kdicts['userbundleKey'][u][bid]
        #except:
        #    bkey = kdicts['userbundleKey'][u][bid] = kdicts['userbundleCtr']
        #    kdicts['userbundleCtr'] += 1
                    
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
        
        if update:
            # if cnt % 1000000 == 0 : gc.collect()
            pdicts['user_id'][ukey, 1] += 1
            pdicts['user_id'][ukey, 4 + part*2 + 1] += 1
            pdicts['content_id'][ckey, 2] += 1
            #pdicts['bundle_id'][bkey, 1] += 1
            pdicts['content_id'][ckey, 0] = yprev + 1
            if yprev: 
                pdicts['user_id'][ukey, 0] += 1
                pdicts['content_id'][ckey, 1] += 1
                #pdicts['bundle_id'][bkey, 0] += 1
                pdicts['user_id'][ukey, 4 + part*2] += 1
            if pexp:
                if yprev: 
                    pdicts['user_id'][ukey, 2] += yprev 
                pdicts['user_id'][ukey, 3] += 1
                
    for t1, (matcu, matascu) in enumerate(zip([cu, expcu, cidcu,], 
                                              [acsu, expacsu, cidacsu])):
        df[f'counts___feat{t1}'] = matcu
        df[f'avgcorrect___feat{t1}'] =  (matascu / (matcu + 0.001)).astype(np.float16)
        #gc.collect()
    del cu, expcu, acsu, expacsu
    for t, i in enumerate(range(1,8)):  
        df[f'counts___feat{t1+t+1}'] = partsdict[i]['cu']
        df[f'avgcorrect___feat{t1+t+1}'] =  (partsdict[i]['acsu']  / \
                                             (partsdict[i]['cu'] + 0.001)).astype(np.float16)
        del partsdict[i]
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
            
            # if cnt % 1000000 == 0 : gc.collect()
            pdicts['user_id'][ukey, 1] += 1
            pdicts['user_id'][ukey, 4 + part*2 + 1] += 1
            pdicts['content_id'][ckey, 2] += 1
            #pdicts['bundle_id'][bkey, 1] += 1
            pdicts['content_id'][ckey, 0] = yprev + 1
            if yprev: 
                pdicts['user_id'][ukey, 0] += 1
                pdicts['content_id'][ckey, 1] += 1
                #pdicts['bundle_id'][bkey, 0] += 1
                pdicts['user_id'][ukey, 4 + part*2] += 1
            if pexp:
                if yprev: 
                    pdicts['user_id'][ukey, 2] += yprev 
                pdicts['user_id'][ukey, 3] += 1

CUT=0
DIR='valfull'
VERSION='V11'
debug = False
validaten_flg = False
FILTCOLS = ['row_id', 'user_id', 'content_id', 'content_type_id',  \
               'answered_correctly', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'user_answer']
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

train[['content_id','user_answer']].groupby('content_id')['user_answer'].nunique().value_counts().reset_index()

# answered correctly average for each content
from scipy.stats import entropy
def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

content_df1 = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df1.columns = ['content_id', 'answered_correctly_avg_c']
content_df2 = train.groupby(['content_id','user_id']).size().reset_index()
content_df2 = content_df2.groupby(['content_id'])[0].mean().astype(np.float16).reset_index()
content_df2.columns = ['content_id', 'attempts_avg_c']
content_df3 = train[['content_id', 'user_id','answered_correctly']].drop_duplicates(keep='first')
content_df3 = content_df3[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean']).astype(np.float16).reset_index()
content_df3.columns = ['content_id', 'answered_correctly_first_avg_c']
content_df4 = train[['content_id','user_answer']].groupby('content_id')['user_answer'].apply(entropy1).astype(np.float16).reset_index()
content_df4.columns = ['content_id', 'entropy']

content_df  = pd.merge(content_df1, content_df2, on = 'content_id')
content_df  = pd.merge(content_df, content_df3, on = 'content_id')
# content_df  = pd.merge(content_df, content_df4, on = 'content_id')

train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")

# part
questions_df['part'] = questions_df['part'].astype(np.uint8)
train = pd.merge(train, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part']], left_on = 'content_id', right_on = 'question_id', how = 'left')



kdicts = getKeys(train)
pdicts = {
          'content_id' : np.zeros( (int(kdicts['usercontentCtr'] *1.2) , 3), dtype= np.uint8),
          #'bundle_id' : np.zeros((int(kdicts['userbundleCtr'] *1.2) , 3), dtype= np.uint16),
          'user_id' : np.zeros((int(kdicts['userCtr'] *1.2) , 40), dtype= np.int16)
          }


train = add_user_feats(train, pdicts, kdicts)
valid = add_user_feats(valid, pdicts, kdicts)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
# answered correctly average for each content
'''
content_df = train[['content_id','prior_question_elapsed_time_mean']].groupby(['content_id']).agg(['mean']).reset_index()
content_df.columns = ['content_id', 'prior_question_elapsed_time_mean_avg_c']
train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")
'''

# use only last 30M training data for limited memory on kaggle env.
#train = train[-30000000:]

TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', \
         'content_id_answered_correctly_prev', 
         'part', 'prior_question_had_explanation', 'prior_question_elapsed_time']
FEATS += [f'counts___feat{i}' for i in range(10)]
FEATS += [f'avgcorrect___feat{i}' for i in range(10)]

    #dro_cols = list(set(train.columns) - set(FEATS))
y_tr = train[TARGET]
y_va = valid[TARGET]
#train.drop(dro_cols, axis=1, inplace=True)
#valid.drop(dro_cols, axis=1, inplace=True)
_=gc.collect()

lgb_train = lgb.Dataset(train[FEATS], y_tr)
lgb_valid = lgb.Dataset(valid[FEATS], y_va)
# del train, y_tr
_=gc.collect()


model = lgb.train(
                    {'objective': 'binary', 'learning_rate': 0.1}, 
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    early_stopping_rounds=10
                )
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)

model.save_model(f'weights/lgb/model_{VERSION}_valfull_cut0_val.pk')


if True:
    joblib.dump(valid, f'data/valfull/cv1_{VERSION}_valid.pk')
    del valid, train
    gc.collect()
    dumpobj(f'data/valfull/cv1_{VERSION}_prior_question_elapsed_time_mean.pk', prior_question_elapsed_time_mean)
    dumpobj(f'data/valfull/cv1_{VERSION}_content_df.pk', content_df)
    dumpobj(f'data/valfull/cv1_{VERSION}_FEATS.pk', FEATS)
    dumpobj(f'data/valfull/cv1_{VERSION}_TARGET.pk', TARGET)
    dumpobj(f'weights/lgb/pdicts_{VERSION}_cut0_val.pk', pdicts)  
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


#dumpobj(f'weights/lgb/kdicts_{VERSION}_cut0_val.pk', kdicts)


######################################

%timeit a = {1:np.array(1, dtype = np.uint32)}
%timeit a['a'].item()

sys.getsizeof(a)
%timeit b = {'b':1}
%timeit b['b']



kdicts = {}
VERSION = 'V10'
for k in ['bundlekey', 'userCtr', 'usercontentCtr', 'userCtr', \
           'userKey', 'userbundleCtr']:
    kdicts[k] = joblib.load(f'data/valfull/kdicts_{k}_{VERSION}_cut0_val.sav')
for k in ['userbundleKeyMat','usercontentKeyMat']:
    kdicts[k[:-3]] = defaultdict(lambda: {})
    mat = np.array(loadobj(f'data/valfull/cv1_{VERSION}_{k}.pk'))
    dumpobj(f'data/valfull/cv1_{VERSION}_{k}.pk', mat.astype(np.uint32))
    for t, row in enumerate(mat): 
        kdicts[k[:-3]][row[0]][row[1]] = t
    del mat
    gc.collect()
        
    
mat = np.array(loadobj(f'data/valfull/cv1_{VERSION}_{k}.pk'))

np.where(mat==(705741139,       128))

sys.getsizeof(mat)
sys.getsizeof(mat.astype(np.uint32))
mat.max()
    
for k, v in kdicts.items():
    print(f'Object {k} size {sys.getsizeof(v)}')
    try:
        joblib.dump(v, f'data/valfull/kdicts_{k}_{VERSION}_cut0_val.sav')  
    except:
        v = dict(v.copy())
        joblib.dump(v, f'data/valfull/kdicts_{k}_{VERSION}_cut0_val.sav')  
        del v
        gc.collect()



'''
Training until validation scores don't improve for 10 rounds
[100]	training's binary_logloss: 0.53721	valid_1's binary_logloss: 0.552055
[200]	training's binary_logloss: 0.535741	valid_1's binary_logloss: 0.550861
[300]	training's binary_logloss: 0.534937	valid_1's binary_logloss: 0.550331
[400]	training's binary_logloss: 0.534334	valid_1's binary_logloss: 0.549985
[500]	training's binary_logloss: 0.53387	valid_1's binary_logloss: 0.549802
Early stopping, best iteration is:
[537]	training's binary_logloss: 0.533704	valid_1's binary_logloss: 0.549731
auc: 0.7652692783096134
'''


'''
Training until validation scores don't improve for 10 rounds
[100]	training's binary_logloss: 0.537146	valid_1's binary_logloss: 0.551928
[200]	training's binary_logloss: 0.535624	valid_1's binary_logloss: 0.550698
[300]	training's binary_logloss: 0.534854	valid_1's binary_logloss: 0.550194
[400]	training's binary_logloss: 0.53429	valid_1's binary_logloss: 0.549914
[500]	training's binary_logloss: 0.533821	valid_1's binary_logloss: 0.549703
[600]	training's binary_logloss: 0.533368	valid_1's binary_logloss: 0.54952
Early stopping, best iteration is:
[688]	training's binary_logloss: 0.533008	valid_1's binary_logloss: 0.549391
auc: 0.7656801332600419
'''


'''
Training until validation scores don't improve for 20 rounds
[100]	training's binary_logloss: 0.537146	valid_1's binary_logloss: 0.551928
[200]	training's binary_logloss: 0.535624	valid_1's binary_logloss: 0.550698
[300]	training's binary_logloss: 0.534854	valid_1's binary_logloss: 0.550194
[400]	training's binary_logloss: 0.53429	valid_1's binary_logloss: 0.549914
[500]	training's binary_logloss: 0.533821	valid_1's binary_logloss: 0.549703
[600]	training's binary_logloss: 0.533368	valid_1's binary_logloss: 0.54952
[700]	training's binary_logloss: 0.532959	valid_1's binary_logloss: 0.549389
[800]	training's binary_logloss: 0.532627	valid_1's binary_logloss: 0.54928
[900]	training's binary_logloss: 0.532299	valid_1's binary_logloss: 0.549191
[1000]	training's binary_logloss: 0.531981	valid_1's binary_logloss: 0.549123
[1100]	training's binary_logloss: 0.53169	valid_1's binary_logloss: 0.549072
[1200]	training's binary_logloss: 0.531346	valid_1's binary_logloss: 0.548961
[1300]	training's binary_logloss: 0.531047	valid_1's binary_logloss: 0.548887
Early stopping, best iteration is:
[1321]	training's binary_logloss: 0.530978	valid_1's binary_logloss: 0.548858
auc: 0.7662888957865834
'''

'''
CUT = 0 
[100]	training's binary_logloss: 0.543969	valid_1's binary_logloss: 0.558445
[200]	training's binary_logloss: 0.542801	valid_1's binary_logloss: 0.557463
[300]	training's binary_logloss: 0.542165	valid_1's binary_logloss: 0.557052
[400]	training's binary_logloss: 0.541707	valid_1's binary_logloss: 0.556819
[500]	training's binary_logloss: 0.541335	valid_1's binary_logloss: 0.55665
Early stopping, best iteration is:
[572]	training's binary_logloss: 0.54109	valid_1's binary_logloss: 0.556547
auc: 0.7573703891529955
auc: 0.7524555286301969
'''

'''
Training until validation scores don't improve for 10 rounds
[100]	training's binary_logloss: 0.539337	valid_1's binary_logloss: 0.544076
[500]	training's binary_logloss: 0.536561	valid_1's binary_logloss: 0.541386
[1000]	training's binary_logloss: 0.535453	valid_1's binary_logloss: 0.540372
[2000]	training's binary_logloss: 0.534337	valid_1's binary_logloss: 0.539475
[3000]	training's binary_logloss: 0.533543	valid_1's binary_logloss: 0.538891
[4100]	training's binary_logloss: 0.5329	valid_1's binary_logloss: 0.538504
Early stopping, best iteration is:
[4158]	training's binary_logloss: 0.532869	valid_1's binary_logloss: 0.538485
auc: 0.7722441861576729
'''

'''
CUT = 0 All
[500]	training's binary_logloss: 0.536427	valid_1's binary_logloss: 0.541271
[1000]	training's binary_logloss: 0.535359	valid_1's binary_logloss: 0.540306
[1500]	training's binary_logloss: 0.534726	valid_1's binary_logloss: 0.539764
[2000]	training's binary_logloss: 0.534233	valid_1's binary_logloss: 0.539395
[2500]	training's binary_logloss: 0.533815	valid_1's binary_logloss: 0.539102
[3000]	training's binary_logloss: 0.533469	valid_1's binary_logloss: 0.538873
[3500]	training's binary_logloss: 0.533115	valid_1's binary_logloss: 0.538645
[4000]	training's binary_logloss: 0.532835	valid_1's binary_logloss: 0.538481
[4500]	training's binary_logloss: 0.532571	valid_1's binary_logloss: 0.538324
Early stopping, best iteration is:
[4843]	training's binary_logloss: 0.532393	valid_1's binary_logloss: 0.538214
'''

'''
CUT = 0 All
[500]	training's binary_logloss: 0.544271	valid_1's binary_logloss: 0.54821
[1000]	training's binary_logloss: 0.543643	valid_1's binary_logloss: 0.547676
[1500]	training's binary_logloss: 0.543257	valid_1's binary_logloss: 0.547402
[2000]	training's binary_logloss: 0.54296	valid_1's binary_logloss: 0.547203
[2500]	training's binary_logloss: 0.542733	valid_1's binary_logloss: 0.547086
[3000]	training's binary_logloss: 0.542517	valid_1's binary_logloss: 0.546981
[3250]	training's binary_logloss: 0.542417	valid_1's binary_logloss: 0.546945
Early stopping, best iteration is:
[3256]	training's binary_logloss: 0.542415	valid_1's binary_logloss: 0.546944
auc: 0.7623517637187441
'''

'''
CUT=1
[100]	training's binary_logloss: 0.546061	valid_1's binary_logloss: 0.563231
Early stopping, best iteration is:
[117]	training's binary_logloss: 0.545951	valid_1's binary_logloss: 0.563118
auc: 0.7425053552702656
'''

