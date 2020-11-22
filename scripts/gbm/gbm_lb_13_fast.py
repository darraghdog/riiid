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
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',100)

pd.set_option('display.width', 1000)
# funcs for user stats with loop
def add_user_feats(df, pdicts, update = True):
    acsu = np.zeros(len(df), dtype=np.uint32)
    cu = np.zeros(len(df), dtype=np.uint32)
    acsb = np.zeros(len(df), dtype=np.uint32)
    cb = np.zeros(len(df), dtype=np.uint32)
    expacsu = np.zeros(len(df), dtype=np.uint32)
    expcu = np.zeros(len(df), dtype=np.uint32)
    cidacsu = np.zeros(len(df), dtype=np.uint32)
    cidcu = np.zeros(len(df), dtype=np.uint32)
    #tcidacsu = np.zeros(len(df), dtype=np.uint32)
    #tcidcu = np.zeros(len(df), dtype=np.uint32)
    #tagstat = np.zeros((len(df), 4), dtype=np.uint32)
    tstamp = np.zeros((len(df), 5), dtype=np.uint32)
    tstavg = np.zeros(len(df), dtype=np.float32)
    #tstprt = np.zeros(len(df), dtype=np.uint32)

    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})
    
    if update:
        itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp']
    else:
        itercols = ['user_id', 'part', 'prior_question_had_explanation', \
                    'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    for cnt,row in enumerate(df[itercols].values ):

        if update:
            u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp = row
        else:
            u, pexp, eltim, cid, tag, tcid, tstmp = row
        bid = bdict[cid]
        newbid = bid == pdicts['track_b'][u]
        
        ucid = f'{u}__{cid}'
        acsu[cnt] = pdicts['answered_correctly_sum_u_dict'][u]
        cu[cnt] = pdicts['count_u_dict'][u]
        acsb[cnt] = pdicts['answered_correctly_sum_b_dict'][u]
        cb[cnt] = pdicts['count_b_dict'][u]
        expacsu[cnt] = pdicts['pexp_answered_correctly_sum_u_dict'][u]
        expcu[cnt] = pdicts['pexp_count_u_dict'][u]
        cidacsu[cnt] = pdicts['content_id_answered_correctly_sum_u_dict'][ucid]
        cidcu[cnt] = pdicts['content_id_count_u_dict'][ucid]
        tstamp[cnt] = [tstmp - pdicts[f'lag_time{i}'][u] for i in [0,1,2,4,9]]
        tstavg[cnt] = pdicts['lag_time_avg'][u]/ (pdicts['count_u_dict'][u]+0.1)
        #tstprt[cnt] = tstmp - pdicts[f'{part}p_lag_time_part'][u] 
        
        #tags = tag.split()
        #tagcts = [pdicts['tag_count_u_dict'][f'{u}_{t}']  for t in tags]
        #tagstat[cnt] = min(tagcts), max(tagcts), np.mean(tagcts), len(tagcts)
        
        partsdict[part]['acsu'][cnt]  = pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        partsdict[part]['cu'][cnt] = pdicts[f'{part}p_count_u_dict'][u]
        otherparts = [p for p in range(1,8) if p != part]
        for p in range(1,8):
            if p == part: continue
            partsdict[p]['cu'][cnt] = pdicts[f'{p}p_count_u_dict'][u]
        '''
        tagct = sum(pdicts['tag_count_u_dict'][f'{u}_{t}'] for t in tags)
        tagls = sum(pdicts['tag_answered_correctly_sum_u_dict'][f'{u}_{t}'] for t in tags) / (tagct+0.01)
        tagavg[cnt] = tagls
        tagcnt[cnt] = tagct
        '''
        if update:
            #for t in tags: pdicts['tag_count_u_dict'][f'{u}_{t}'] += 1
            pdicts['count_u_dict'][u] += 1
            for i in list(range(1, 10))[::-1]:
                pdicts[f'lag_time{i}'][u] = pdicts[f'lag_time{i-1}'][u] 
            pdicts['lag_time0'][u] = tstmp
            pdicts['lag_time_avg'][u] += tstmp
            pdicts['count_c_dict'][cid] += 1
            pdicts[f'{part}p_count_u_dict'][u] += 1
            #pdicts[f'{part}p_lag_time_part'][u] = tstmp
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
    #df['lag_content_avgtime'] = tstprt
    #df[['tagct_min', 'tagct_max', 'tagct_mean', 'tag_len']] = tagstat
    del cu, expcu, acsu, expacsu
    for t, i in enumerate(range(1,8)):  
        df[f'counts___feat{t1+t+1}'] = partsdict[i]['cu']
        df[f'avgcorrect___feat{t1+t+1}'] =  (partsdict[i]['acsu']  / (partsdict[i]['cu'] + 0.001)).astype(np.float16)
        del partsdict[i]
        #gc.collect()
    #for t, m in  enumerate([tagmin, tagmax, tagavg, tagcnt]):
    #    df[f'tag_correct_stat{t}'] = m
    
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
DIR='val' # valfull
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

# Treat tags same as content, but but just loop update 
# Carry the 

questions_df = pd.read_csv('data/questions.csv')
bdict = questions_df.set_index('question_id')['bundle_id'].to_dict()

train = train.loc[train.content_type_id == False].reset_index(drop=True)
valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

# changing dtype to avoid lightgbm error
train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
valid['prior_question_had_explanation'] = valid.prior_question_had_explanation.fillna(False).astype('int8')

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
          'content_id_lag' : defaultdict(int), 
          'pexp_answered_correctly_sum_u_dict' : defaultdict(int),
          'pexp_count_u_dict': defaultdict(int)}

for i in range(10): 
    pdicts[f'lag_time{i}'] =   defaultdict(int)
for p in train.part.unique():
    pdicts[f'{p}p_answered_correctly_sum_u_dict'] =  defaultdict(int)
    pdicts[f'{p}p_count_u_dict'] =  defaultdict(int)
    #pdicts[f'{p}p_lag_time_part'] =  defaultdict(int)

train = add_user_feats(train, pdicts)
valid = add_user_feats(valid, pdicts)

# fill with mean value for prior_question_elapsed_time
# note that `train.prior_question_elapsed_time.mean()` dose not work!
# please refer https://www.kaggle.com/its7171/can-we-trust-pandas-mean for detail.
prior_question_elapsed_time_mean = train.prior_question_elapsed_time.dropna().values.mean()
train['prior_question_elapsed_time_mean'] = train.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)
valid['prior_question_elapsed_time_mean'] = valid.prior_question_elapsed_time.fillna(prior_question_elapsed_time_mean)

# answered correctly average for each content

# use only last 30M training data for limited memory on kaggle env.
#train = train[-30000000:]

TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', 'cid_answered_correctly', \
         'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', \
         'answered_correctly_ct_c', 'answered_correctly_last_avg_c', 'lag_content_avgtime']
FEATS += [f'counts___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
FEATS += [f'lag_content_time{i}' for i in [0,1,2,5,10]]
#FEATS += ['tagct_min', 'tagct_max', 'tagct_mean', 'tag_len']
#FEATS += [f'tag_correct_stat{i}' for i in range(4)]

    
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
                    early_stopping_rounds=20
                )
print('auc:', roc_auc_score(y_va, model.predict(valid[FEATS])))
_ = lgb.plot_importance(model)

#### New 
#- Add all the parts
#- Multiple lags

# Try storing each day that the user is in, and how many questions per day


model.save_model(f'weights/lgb/model_{VERSION}_valfull_cut0_val.pk')
model1 = lgb.Booster(model_file=f'weights/lgb/model_{VERSION}_valfull_cut0_val.pk' )
_ = lgb.plot_importance(model1)

train[['user_id','answered_correctly','content_type_id']]

dir(lgb)


'''
Training until validation scores don't improve for 20 rounds
[100]	training's binary_logloss: 0.528426	valid_1's binary_logloss: 0.541937
[200]	training's binary_logloss: 0.525694	valid_1's binary_logloss: 0.539477
[300]	training's binary_logloss: 0.524243	valid_1's binary_logloss: 0.538392
[400]	training's binary_logloss: 0.52322	valid_1's binary_logloss: 0.537735
[500]	training's binary_logloss: 0.52242	valid_1's binary_logloss: 0.537343
[600]	training's binary_logloss: 0.521785	valid_1's binary_logloss: 0.537053
[700]	training's binary_logloss: 0.521145	valid_1's binary_logloss: 0.536742
[800]	training's binary_logloss: 0.520573	valid_1's binary_logloss: 0.536526
Early stopping, best iteration is:
[835]	training's binary_logloss: 0.520387	valid_1's binary_logloss: 0.536437
auc: 0.7804127321104475
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

'''
Training until validation scores don't improve for 20 rounds
[100]	training's binary_logloss: 0.537261	valid_1's binary_logloss: 0.552061
[200]	training's binary_logloss: 0.535783	valid_1's binary_logloss: 0.550912
[300]	training's binary_logloss: 0.534999	valid_1's binary_logloss: 0.55039
[400]	training's binary_logloss: 0.534394	valid_1's binary_logloss: 0.550079
[500]	training's binary_logloss: 0.533939	valid_1's binary_logloss: 0.549887
[600]	training's binary_logloss: 0.533532	valid_1's binary_logloss: 0.549746
[700]	training's binary_logloss: 0.533163	valid_1's binary_logloss: 0.549629
[800]	training's binary_logloss: 0.532799	valid_1's binary_logloss: 0.549547
[900]	training's binary_logloss: 0.532464	valid_1's binary_logloss: 0.549483
[1000]	training's binary_logloss: 0.532112	valid_1's binary_logloss: 0.549413
Early stopping, best iteration is:
[988]	training's binary_logloss: 0.53214	valid_1's binary_logloss: 0.549407
auc: 0.7656682620630719
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

