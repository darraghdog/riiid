# https://www.kaggle.com/takamotoki/lgbm-iii-part3-adding-lecture-features/data
import os
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import datatable as dt
PATH = '/Users/dhanley/Documents/riiid'
os.chdir(PATH)


'''
When we commit vs. when we submit a notebook

if the questions.csv contains the same question_id.
if the lectures.csv contains the same question_id.
if all the question ids in the private test dataset are seen in train.csv and questions.csv.
if all the lecture ids in the private test dataset are seen in train.csv and lectures.csv.
if a batch from the private test dataset has timestamps larger (or at least equal) than the timestamps of the corresponding users in train.csv.
if the batches from the private test datasets have monotonically increasing (actually, non-decreasing) timestamps.
if there is only one (if any) question bundle for a sinlge user in a single test batch (this is true as mentioned in the competition Data page).
if each question bundle is in a consecutive block in a test batch (despite the row ids may jumps).
if the question bundle for a single user in a test batch will always be at the end of this user's sequence.
'''


'''
train = pd.read_csv('data/train.csv',
                   usecols=[1, 2, 3, 4, 5, 7, 8, 9],
                   dtype={'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )
train.to_feather('data/train.gbm.feather')
'''
train = pd.read_feather('data/train.gbm.feather')
test = pd.read_csv('data/example_test.csv',
                   usecols=[1, 2, 3, 4, 5, 7, 8, 9],
                   dtype={'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'})

train.groupby('user_id').timestamp.min().hist()
test.groupby('user_id').timestamp.min().hist(bins = 100)
train.iloc[0]


train.prior_question_had_explanation.value_counts()
questions_df = pd.read_csv('data/questions.csv',
                            usecols=[0, 3],
                            dtype={'question_id': 'int16',
                              'part': 'int8'})
lectures_df = pd.read_csv('data/lectures.csv')
lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')
lectures_df = pd.get_dummies(lectures_df, columns=['part', 'type_of'])
part_lectures_columns = lectures_df.filter(like='part').columns.tolist()
types_of_lectures_columns = lectures_df.filter(like='type_of_').columns.tolist()

lectures_df.head()

# merge lecture features to train dataset
train_lectures = train[train.content_type_id == True].merge(lectures_df, 
                                                            left_on='content_id', 
                                                            right_on='lecture_id', 
                                                            how='left')
train_lectures.head()

# collect per user stats
filt = part_lectures_columns + types_of_lectures_columns
user_lecture_stats_part = train_lectures.groupby('user_id')[filt].sum()
user_lecture_stats_part.head()


# add boolean features
newcols = user_lecture_stats_part.add_suffix('_boolean').columns.tolist()
user_lecture_stats_part[newcols] = (user_lecture_stats_part > 0).astype(int)

#clearing memory
del(train_lectures)
gc.collect()

#removing True or 1 for content_type_id
train = train[train.content_type_id == False].sort_values('timestamp').reset_index(drop = True)

train[(train.content_type_id == False)].task_container_id.nunique()

#saving value to fillna
elapsed_mean = train.prior_question_elapsed_time.mean()

# Average questions seen per container and cumulative quetions. 
group1 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['count'])
group1.columns = ['avg_questions']
group2 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['nunique'])
group2.columns = ['avg_questions']
group3 = group1 / group2

group3['avg_questions_seen'] = group3.avg_questions.cumsum()
group3.iloc[0].avg_questions_seen


# Average answered correctly. 
results_u_final = train.loc[train.content_type_id == False, ['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_final.columns = ['answered_correctly_user']
results_u_final.answered_correctly_user.describe()
results_u2_final = train.loc[train.content_type_id == False, ['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_final.columns = ['explanation_mean_user']
results_u2_final.explanation_mean_user.describe()


# Merging
train = pd.merge(train, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')

results_q_final = train.loc[train.content_type_id == False, ['question_id','answered_correctly']].groupby(['question_id']).agg(['mean'])
results_q_final.columns = ['quest_pct']

results_q2_final = train.loc[train.content_type_id == False, ['question_id','part']].groupby(['question_id']).agg(['count'])
results_q2_final.columns = ['count']

question2 = pd.merge(questions_df, results_q_final, left_on = 'question_id', right_on = 'question_id', how = 'left')

question2 = pd.merge(question2, results_q2_final, left_on = 'question_id', right_on = 'question_id', how = 'left')

question2.quest_pct = round(question2.quest_pct,5)


display(question2.head(), question2.tail())

train.head()

prior_mean_user = results_u2_final.explanation_mean_user.mean()
train.drop(['timestamp', 'content_type_id', 'question_id', 'part'], axis=1, inplace=True)


# Validation set
validation = train.groupby('user_id').tail(5)
train = train[~train.index.isin(validation.index)]

results_u_val = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_val.columns = ['answered_correctly_user']

results_u2_val = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_val.columns = ['explanation_mean_user']


# Extracting training data
X = train.groupby('user_id').tail(18)
train = train[~train.index.isin(X.index)]
len(X) + len(train) + len(validation)

X.answered_correctly.mean()
train.answered_correctly.mean()

results_u_X = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_X.columns = ['answered_correctly_user']

results_u2_X = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_X.columns = ['explanation_mean_user']

# Merging Data

X = pd.merge(X, group3, left_on=['task_container_id'], right_index= True, how="left")
X = pd.merge(X, results_u_X, on=['user_id'], how="left")
X = pd.merge(X, results_u2_X, on=['user_id'], how="left")

X = pd.merge(X, user_lecture_stats_part, on=['user_id'], how="left")


validation = pd.merge(validation, group3, left_on=['task_container_id'], right_index= True, how="left")
validation = pd.merge(validation, results_u_val, on=['user_id'], how="left")
validation = pd.merge(validation, results_u2_val, on=['user_id'], how="left")

validation = pd.merge(validation, user_lecture_stats_part, on=['user_id'], how="left")


len(range(0))

for epoch in range(0):
    print(epoch)


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

X.prior_question_had_explanation.fillna(False, inplace = True)
validation.prior_question_had_explanation.fillna(False, inplace = True)

validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])
X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])


content_mean = question2.quest_pct.mean()

question2.quest_pct.mean()
#there are a lot of high percentage questions, should use median instead?
#filling questions with no info with a new value
question2.quest_pct = question2.quest_pct.mask((question2['count'] < 3), .65)
#filling very hard new questions with a more reasonable value
question2.quest_pct = question2.quest_pct.mask((question2.quest_pct < .2) & (question2['count'] < 21), .2)
#filling very easy new questions with a more reasonable value
question2.quest_pct = question2.quest_pct.mask((question2.quest_pct > .95) & (question2['count'] < 21), .95)


X = pd.merge(X, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
validation = pd.merge(validation, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
X.part = X.part - 1
validation.part = validation.part - 1


y = X['answered_correctly']
X = X.drop(['answered_correctly'], axis=1)
X.head()
y_val = validation['answered_correctly']
X_val = validation.drop(['answered_correctly'], axis=1)


cols = ['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
       'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
       'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter',
       'part_1_boolean', 'part_2_boolean', 'part_3_boolean', 'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean',
       'type_of_concept_boolean', 'type_of_intention_boolean', 'type_of_solving_question_boolean', 'type_of_starter_boolean']
X = X[cols]
X_val = X_val[cols]



def dffillna(X):
    # Filling with 0.5 for simplicity; there could likely be a better value
    X['answered_correctly_user'].fillna(0.65,  inplace=True)
    X['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
    X['quest_pct'].fillna(content_mean, inplace=True)
    
    X['part'].fillna(4, inplace = True)
    X['avg_questions_seen'].fillna(1, inplace = True)
    X['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
    X['prior_question_had_explanation_enc'].fillna(0, inplace = True)
    
    cols = ['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6',
           'part_7', 'part_1_boolean', 'part_2_boolean', 'part_3_boolean',
           'part_4_boolean', 'part_5_boolean', 'part_6_boolean', 'part_7_boolean']
    cols += ['type_of_concept', 'type_of_intention', 'type_of_solving_question',
           'type_of_starter', 'type_of_concept_boolean',
           'type_of_intention_boolean', 'type_of_solving_question_boolean',
           'type_of_starter_boolean']
    for c in cols:
        X[c].fillna(0, inplace = True)
    return X

X = dffillna(X)
Xval = dffillna(Xval)


import lightgbm as lgb

params = {
    'objective': 'binary',
    'boosting' : 'gbdt',
    'max_bin': 800,
    'learning_rate': 0.0175,
    'num_leaves': 80
}

lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation_enc'])
lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation_enc'], reference=lgb_train)


model = lgb.train(
    params, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=50,
    num_boost_round=10000,
    early_stopping_rounds=12
)

dir(model)

os.getcwd()

model.save_model('weights/lgb/model_test.pk')
mod1 = lgb.Booster(model_file='weights/lgb/model_test.pk' )














