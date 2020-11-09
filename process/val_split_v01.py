# https://www.kaggle.com/takamotoki/lgbm-iii-part3-adding-lecture-features/data
import os
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import datatable as dt
PATH = '/Users/dhanley/Documents/riiid'
os.chdir(PATH)
from scripts.utils import Iter_Valid


VALRATIO = 0.06
# Break up val
NEWUSERS=0.1
CAPRATIO=0.3

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

userct = train.groupby('user_id').content_id.count()
userct.shape
userct.clip(0, 200).hist(bins = 100)

'''
Split users to ~ 10 of the data for validating
Validation set ~ 39365 users
'''
np.expand_dims(train.user_id.unique(), 0).shape
val_users = pd.Series(train.user_id.unique()).sample(frac=VALRATIO, replace=False, random_state=0).sort_values().values
print(val_users.shape)
allval = train.set_index('user_id').loc[val_users].reset_index()

'''
from this train set split off ~10% as new users, 
randomly take between 10% and 30% of the final questions from the rest
'''
allval['train'] = True
new_users = pd.Series(allval.user_id.unique()).sample(frac=NEWUSERS, replace=False, random_state=0).sort_values().values
allval['train'][allval.user_id.isin(new_users)] = False
# Sort by user, timestamp
sortcols = ['user_id', 'timestamp']
allval = allval.sort_values(sortcols).reset_index(drop=True)
allval['seq'] = allval.groupby('user_id').cumcount()
allval['count'] = allval.groupby('user_id')["content_id"].transform("count") - 1
# Make randomly distribute 
testsamp = pd.DataFrame({'user_id': allval.user_id.unique(),
              'sampratio' : np.random.choice(int(CAPRATIO * 100), 
                                                size=len(allval.user_id.unique()), 
                                                replace=True) / 100  })
allval = allval.merge(testsamp, on = 'user_id', how = 'left').sort_values(sortcols)
# Take the last K ratio of each user and put in test
allval['train'] = ((1-allval['seq']/allval['count']) > allval['sampratio'])
# Now split into the new dataset
trnsub = allval[ train.columns.tolist()][allval['train'] .values].reset_index(drop = True)
valsub = allval[ train.columns.tolist()][~allval['train'] .values].reset_index(drop = True)
# write out to disk
trnsub .to_feather('data/val/trainsub.gbm.feather')
valsub .to_feather('data/val/validsub.gbm.feather')



