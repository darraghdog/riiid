# https://www.kaggle.com/its7171/cv-strategy

import os
import gc
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import datatable as dt
PATH = '/Users/dhanley/Documents/riiid'
os.chdir(PATH)
from scripts.utils import Iter_Valid, dumpobj, loadobj
random.seed(1)


VALRATIO = 0.1
val_size = 250000

'''
The hidden test set contains new users but not new questions.
The test data follows chronologically after the train data. The test iterations give interactions of users chronologically.
Each group will contain interactions from many different users, but no more than one task_container_id of questions from any single user. 
Each group has between 1 and 1000 users.
Expect to see roughly 2.5 million questions in the hidden test set.
The API will also consume roughly 15 minutes of runtime for loading and serving the data.
The API loads the data using the types specified in Data Description page.
'''



'''
train = pd.read_csv('data/train.csv',
                   dtype={'row_id': 'int64',
                          'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'user_answer': 'int8',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )
train.to_feather('data/train.gbm.feather')
'''
train = pd.read_feather('data/train.gbm.feather')

'''
Split users to ~ 10% of the data for validating
Validation set ~ 39365 users
'''
np.expand_dims(train.user_id.unique(), 0).shape
val_users = pd.Series(train.user_id.unique()).sample(frac=VALRATIO, replace=False, random_state=0).sort_values().values
print(val_users.shape)
train = train[train.user_id.isin(val_users)]
# train = train.set_index('user_id').loc[val_users].reset_index()

# Now follow = https://www.kaggle.com/its7171/cv-strategy

max_timestamp_u = train[['user_id','timestamp']].groupby(['user_id']).agg(['max']).reset_index()
max_timestamp_u.columns = ['user_id', 'max_time_stamp']
MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

def rand_time(max_time_stamp):
    interval = MAX_TIME_STAMP - max_time_stamp
    rand_time_stamp = random.randint(0,interval)
    return rand_time_stamp

max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(rand_time)
train = train.merge(max_timestamp_u, on='user_id', how='left')
train['viretual_time_stamp'] = train.timestamp + train['rand_time_stamp']
train = train.sort_values(['viretual_time_stamp', 'row_id']).reset_index(drop=True)


for cv in range(5):
    valid = train[-val_size:].reset_index(drop=True)
    train = train[:-val_size].reset_index(drop=True)
    # check new users and new contents
    new_users = len(valid[~valid.user_id.isin(train.user_id)].user_id.unique())
    valid_question = valid[valid.content_type_id == 0]
    train_question = train[train.content_type_id == 0]
    
    new_contents = len(valid_question[~valid_question.content_id.isin(train_question.content_id)].content_id.unique())    
    print(f'cv{cv} {train_question.answered_correctly.mean():.3f} {valid_question.answered_correctly.mean():.3f} {new_users} {new_contents}')
    valid.to_feather(f'data/val/cv{cv+1}_valid.feather')
    train.to_feather(f'data/val/cv{cv+1}_train.feather')
    iter_valid = Iter_Valid(valid,max_user=10**6)
    dumpobj(f'data/val/iter{cv+1}_valid.pk', iter_valid)

'''
cv0 0.659 0.642 1076 0
cv1 0.660 0.653 1054 0
cv2 0.660 0.656 973 0
cv3 0.660 0.662 916 0
cv4 0.660 0.654 860 0
'''


predicted = []
def set_predict(df):
    predicted.append(df)
    
iter_test = iter_valid
pbar = tqdm(total=250000)
previous_test_df = None
for (current_test, current_prediction_df) in iter_test:
    if previous_test_df is not None:
        answers = eval(current_test["prior_group_answers_correct"].iloc[0])
        responses = eval(current_test["prior_group_responses"].iloc[0])
        previous_test_df['answered_correctly'] = answers
        previous_test_df['user_answer'] = responses
        # your feature extraction and model training code here
    previous_test_df = current_test.copy()
    current_test = current_test[current_test.content_type_id == 0]
    # your prediction code here
    current_test['answered_correctly'] = 0.5
    set_predict(current_test.loc[:,['row_id', 'answered_correctly']])
    pbar.update(len(current_test))