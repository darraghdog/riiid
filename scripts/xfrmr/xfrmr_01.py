# https://www.kaggle.com/takamotoki/lgbm-iii-part3-adding-lecture-features/data
import os
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import datatable as dt
from tqdm import tqdm
PATH = '/Users/dhanley/Documents/riiid'
os.chdir(PATH)
from scripts.utils import Iter_Valid, dumpobj, loadobj
import lightgbm as lgb
from sklearn.metrics import log_loss, auc, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
#pd.options.display.max_rows = 100

'''
Excercise empbeddings : 
• Exercise ID: A latent vector is assigned to an ID unique to
each exercise.
• Exercise category: Each exercise belongs to a category of
the domain subject. A latent vector is assigned to each category.
• Position: The position (1st, 2nd, ...) of an exercise or a
response in the input sequence is represented as a position
embedding vector. The position embeddings are shared
across the exercise sequence and the response sequence.
Response empeddings
• Response: A latent vector is assigned to each possible value
(0 or 1) of a student’s response ri.
• Elapsed time: The time a student took to respond in seconds
is rounded to an integer value. A latent vector is assigned to
each integer between 0 and 300, inclusive. Any time more
than 300 seconds is capped off to 300 seconds.
• Timestamp: Month, day and hour of the absolute time when
a student received each exercise is recorded. A unique latent
vector is assigned for every possible combination of month,
day and hour.

Log loss for cut0 : 0.5840 
AUC for cut0      : 0.7217 

'''

excl = ['max_time_stamp' , 'rand_time_stamp', 'viretual_time_stamp']
ecols = ['question_id', 'bundle_id', 'part']
rcols = ['answered_correctly', 'elapsedtime' ]

CUT = 0
ITER = False
valid = pd.read_feather(f'data/val/cv{CUT+1}_valid.feather')
train = pd.read_feather(f'data/val/cv{CUT+1}_train.feather')
iter_test = loadobj(f'data/val/iter{CUT+1}_valid.pk')

keepcols = [c for c in train.columns if c not in excl]
valid = valid [keepcols]
train = train [keepcols]

# Make an alldf using the valid data
yvalact = valid[['row_id', 'answered_correctly']]
valid['answered_correctly'] = np.nan # -2 for 
alldf = pd.concat([train, valid], 0)
del train, valid
gc.collect()


# Load up the questions and lectures
questions_df = pd.read_csv('data/questions.csv',
                            dtype={'question_id': 'int16',
                              'part': 'int8'})
lectures_df = pd.read_csv('data/lectures.csv')
questions_df = questions_df.rename(columns = {'part': 'qpart', 'tags': 'qtags'})
lectures_df = lectures_df.rename(columns = {'part': 'lpart', 'tag': 'ltag'})
lectures_df['type_of'] = lectures_df['type_of'].replace('solving question', 'solving_question')

# merge lecture features to train dataset
alldf_lectures = alldf[alldf.content_type_id == True].merge(lectures_df, 
                                                            left_on='content_id', 
                                                            right_on='lecture_id', 
                                                            how='left')
alldf_lectures.head()


# Lets remove lectures for now...
alldf = alldf[alldf.content_type_id == False].sort_values('timestamp').reset_index(drop = True)

# Join questions 
alldf = pd.merge(alldf, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')

# Get lag 
alldf = alldf.sort_values(['user_id', 'timestamp'], 0)
# This may be elapsed time
alldf['elapsedtime'] = alldf.timestamp - alldf.timestamp.shift(1)
alldf['elapsedtime'][alldf.user_id != alldf.user_id.shift(1)] = np.nan
# Like the paper, create an embedding for first 300 seconds and add one for unknown.
alldf['elapsedtime'] = (alldf['elapsedtime']/1000) \
        .round().clip(0, 300) \
        .fillna(-1) \
        .astype(np.int16)



alldf.iloc[0]
















#removing True or 1 for content_type_id
train = train[train.content_type_id == False].sort_values('timestamp').reset_index(drop = True)
valid = valid[valid.content_type_id == False].sort_values('timestamp').reset_index(drop = True)

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








