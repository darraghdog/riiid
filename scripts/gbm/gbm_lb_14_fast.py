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
pd.set_option('display.max_rows',100)

pd.set_option('display.width', 1000)

def hrfn(val):
    return int(round(val/(3600*1000) % 24))
def dayfn(val):
    return int(round(val/(3600*1000*24) % 7))

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
    tstamp = np.zeros((len(df), 5), dtype=np.uint32)
    tstavg = np.zeros(len(df), dtype=np.float32)
    ctunq = np.zeros((len(df), 2), dtype=np.uint32)
    lectct = np.zeros((len(df), 7), dtype=np.uint32)
    lectavg = np.zeros((len(df), 3), dtype=np.float32)
    pexpm = np.zeros((len(df), 1), dtype=np.uint8)
    contid = np.zeros((len(df), 1), dtype=np.uint8)

    partsdict = defaultdict(lambda : {'acsu' : np.zeros(len(df), dtype=np.uint32),
                                      'cu' : np.zeros(len(df), dtype=np.uint32)})

    itercols = ['user_id','answered_correctly', 'part', \
                    'prior_question_had_explanation', 'prior_question_elapsed_time', 'content_id', 'tags', \
                        'task_container_id', 'timestamp', 'content_type_id']
    if not update:
        itercols = [f for f in itercols if f!='answered_correctly']
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].fillna(False).astype(np.uint8)
    
    for cnt,row in enumerate(tqdm(df[itercols].values, total = df.shape[0]) ):
        if update:
            u, yprev, part, pexp, eltim, cid, tag, tcid, tstmp, ctype = row
        else:
            u, pexp, eltim, cid, tag, tcid, tstmp, ctype = row
        
        tags = [int(t)  for t in tag.split()]
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
        
        
        bid = bdict[cid]
        newbid = bid == pdicts['track_b'][u]
        
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
        cu[cnt] = pdicts['count_u_dict'][u]
        acsb[cnt] = pdicts['answered_correctly_sum_b_dict'][u]
        cb[cnt] = pdicts['count_b_dict'][u]
        expacsu[cnt] = pdicts['pexp_answered_correctly_sum_u_dict'][u]
        expcu[cnt] = pdicts['pexp_count_u_dict'][u]
        cidacsu[cnt] = pdicts['content_id_answered_correctly_sum_u_dict'][ucid]
        cidcu[cnt] = pdicts['content_id_count_u_dict'][ucid]
        tstamp[cnt] = [tstmp - pdicts[f'lag_time{i}'][u] for i in [0,1,2,4,9]]
        tstavg[cnt] = pdicts['lag_time_avg'][u]/ (pdicts['count_u_dict'][u]+0.1)
        ctunqsum = len(pdicts['ctunique'][uiddict[u]])
        ctunq[cnt] = ctunqsum, 1000*(ctunqsum / (pdicts['count_u_dict'][u] + 0.01))
        
        
        partsdict[part]['acsu'][cnt]  = pdicts[f'{part}p_answered_correctly_sum_u_dict'][u]
        partsdict[part]['cu'][cnt] = pdicts[f'{part}p_count_u_dict'][u]
        for p in range(1,8):
            if p == part: continue
            partsdict[p]['cu'][cnt] = pdicts[f'{p}p_count_u_dict'][u]
            partsdict[p]['acsu'][cnt]  = pdicts[f'{p}p_answered_correctly_sum_u_dict'][u]
            
        if update:
            pdicts['count_u_dict'][u] += 1
            pdicts['ctunique'][uiddict[u]].add(cid)
            
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
    df[['ctunique_sum', 'ctunique_attempt_ration']] = ctunq
    df[[f'lecture_stats_{i}' for i in range(7)]] = lectct
    df[[f'lecture_stats_{i}' for i in range(7,10)]] = lectavg
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
                   'prior_question_had_explanation', 'task_container_id', 'timestamp']
valid = pd.read_feather(f'data/{DIR}/cv{CUT+1}_valid.feather')[FILTCOLS]
train = pd.read_feather(f'data/{DIR}/cv{CUT+1}_train.feather')[FILTCOLS]

# Treat tags same as content, but but just loop update 
# Carry the 

questions_df = pd.read_csv('data/questions.csv')
ldf = pd.read_csv('data/lectures.csv')
ldf.type_of = ldf.type_of.str.replace(' ', '_')
ldict = ldf.set_index('lecture_id').to_dict()
lecture_types = [t for t in ldf.type_of.unique() if t!= 'starter']

adf = pd.merge(valid.query('content_type_id == 1'), ldf, left_on='content_id', right_on = 'lecture_id', how = 'left')

adf[['user_id', 'type_of']].groupby('type_of')['user_id'].count()

train.user_id.drop_duplicates().isin

bdict = questions_df.set_index('question_id')['bundle_id'].to_dict()

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

train = pd.merge(train, content_df, on=['content_id'], how="left")
valid = pd.merge(valid, content_df, on=['content_id'], how="left")


# part
questions_df['part'] = questions_df['part'].astype(np.uint8)
train = pd.merge(train, questions_df[['question_id', 'part', 'tags']], left_on = 'content_id', right_on = 'question_id', how = 'left')
valid = pd.merge(valid, questions_df[['question_id', 'part', 'tags']], left_on = 'content_id', right_on = 'question_id', how = 'left')

train[['part', 'question_id']] = train[['part', 'question_id']].fillna(0).astype(np.int16)
valid[['part', 'question_id']] = valid[['part', 'question_id']].fillna(0).astype(np.int16)


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
uiddict = dict((u,t) for t,u in \
               enumerate(set(train.user_id.unique().tolist()+valid.user_id.unique().tolist())))
nuser = max([k for k in uiddict.values()])

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
          'ctunique' : [set() for i in range(nuser+1)],
          'uiddict' : uiddict,
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

train.shape
train[[f'lecture_stats_{i}' for i in range(0,10)]].sample(11)

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

TARGET = 'answered_correctly'
FEATS = ['answered_correctly_avg_c', 'attempts_avg_c', 'answered_correctly_first_avg_c', 'cid_answered_correctly', \
         'part', 'prior_question_had_explanation', 'prior_question_elapsed_time', \
         'answered_correctly_ct_c', 'answered_correctly_last_avg_c', 'lag_content_avgtime']
FEATS += [f'counts___feat{i}' for i in range(11)]
FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
#FEATS += [f'avgcorrect___feat{i}' for i in range(11)]
FEATS += [f'lag_content_time{i}' for i in [0,1,2,5,10]]
FEATS += ['ctunique_sum', 'ctunique_attempt_ration']
FEATS += [f'lecture_stats_{i}' for i in range(10)]
#FEATS += [f'lda_comp{i}' for i in range(5)]
FEATS += [f'tag{i}' for i in range(6)]
FEATS += ['content_id']
#FEATS += [f'lecture_stats_type_{l}' for l in range(9)]

y_tr = train[TARGET]
y_va = valid[TARGET]
_=gc.collect()

categoricals = ['part', 'content_id', 'user_id_le'] + [f'tag{i}' for i in range(6)]
lgb_train = lgb.Dataset(train[FEATS], y_tr, categorical_feature = categoricals)
lgb_valid = lgb.Dataset(valid[FEATS], y_va, categorical_feature = categoricals)
_=gc.collect()

model = lgb.train(
                    {'objective': 'binary', 
                     'min_data_per_group':5000,
                     'learning_rate': 0.1},
                    lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    num_boost_round=10000,
                    early_stopping_rounds=40,
                )
print('auc:', roc_auc_score(y_va.values, model.predict(valid[FEATS])  )  )
_ = lgb.plot_importance(model, max_num_features = 50, figsize = (5,10))

#### Try 
#- Any way to do nunique questions - like a boolean array / sparse array
#- Counts 
# if - 2(30 samples), 3(25 samples), 5(20 samples), 7(15 samples), 8(10 samples), 
#- nunique_counts is 5, nunique_counts_ratio is 0.02, top_counts is 30, top_counts_ratio is 0.3,
# - start with lectures... time since lecture etc. 

#### New 
#- Ctunique
#- Remove duplicated feats :( in FEATS)
#- Lectures
# Add tags and content_id as categorical with a min_data_per_group
#- Look at these features : https://www.kaggle.com/calebeverett/riiid-submit : https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475
# - LDA topics of IPs related to app.
# - LDA over tags
# ad task_container_id seq

# Target 0.7868

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

