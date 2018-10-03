# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:54:39 2018

@author: Brian and Abubaker
"""
import numpy as np 
import pandas as pd 
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


df = pd.read_csv(r'pictures-train.tsv', sep = '\t')


'''
Data Cleaning
'''
# Change etitle, region to categorical. Date columns to DateTime

df['votedon'] = pd.to_datetime(df['votedon'], format='%Y-%m-%d')

# Change negatives to NA

df.loc[df.votes < 0] = np.nan
df.loc[df.viewed < 0] = np.nan
df.loc[df.n_comments < 0] = np.nan

# Drop rows where indendent variable (votes) is NaN
df = df.dropna(axis=0, subset=['votes']) 

# check how many NaNs are in each feature
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# very few missing values, delete those rows
df = df.dropna(how='any')

#takenon column contains many instances where month or day is unknown (00)
# we will convert all these instances to 01 and convert to DateTime

df['takenon'] = df['takenon'].str.replace('-00', '-01')
df['takenon'] = pd.to_datetime(df['takenon'], format='%Y-%m-%d')

'''
Engineer new features
'''
# does the day of the week a picture gets posted effect the votes? lets
# extract day of week and create a weekday/weekend from votedon feature

df['day_of_week'] = df['votedon'].dt.weekday_name
df['day_of_week'] = ((pd.DatetimeIndex(df['votedon']).dayofweek) // 4 == 1).astype(float)
df['day_of_week'] = np.where(df['day_of_week']==1, 'weekend', 'weekday')

# use month and year as categorical feature
df['MonthPosted'] = df['votedon'].dt.month.astype(object)
df['YearPosted'] = df['votedon'].dt.year.astype(object)


# feature engineer 'author_range' column
# author_range' is the time between an authors first and latest post
# change from timedelta to float

ActivityRange = df.groupby('author_id')['votedon'].agg(['max','min']).reset_index()
ActivityRange['author_range'] = (ActivityRange['max']-ActivityRange['min'])

df = pd.merge(df,ActivityRange[['author_id','author_range']], how='left', on='author_id')

df['author_range'] = df['author_range'].dt.days

# feature engineer 'image_age' columns
#'author_range' is the time between votedon and takenon (estimate)
# change from timedelta to float
df['image_age'] = abs(df['votedon']-df['takenon'])
df['image_age'] = df['image_age'].dt.days

# feature engineer post_age
df['post_age'] = (max(df['votedon'])- df['votedon']).dt.days

# log of independent variable to fit normal distrubution
df['votes_log'] = np.log(df['votes'].replace(0,1))


'''
Exploratoy Data Analysis
'''
## Distributions of of posts by Category and Region

plt.figure(1, figsize=(10,10))
plt.title("Post Category", y=1)
df['etitle'].value_counts().plot.pie()
plt.savefig("Post Category.png", dpi=200)
plt.show()


plt.figure(2, figsize=(10,10))
plt.title("Post Region", y=1)
df['region'].value_counts().plot.pie()
plt.savefig("Post Region.png", dpi=200)
plt.show()


## Histograms of Upvotes, views, comments

plt.figure(3, figsize=(10,10))
plt.title('Upvotes Distribution')
plt.xlabel('Upvotes')
plt.hist(df['votes'].dropna(), bins = 60,  label = 'Upvotes')
plt.legend()
plt.savefig("Upvotes Distribution.png", dpi=200)
plt.show()


plt.figure(4, figsize=(10,10))
plt.title('Views Distribution')
plt.xlabel('Post Views')
plt.hist(df['viewed'].dropna(), bins = 30, label = 'Views')
plt.legend()
plt.savefig("Views Distribution.png", dpi=200)
plt.show()


plt.figure(5, figsize=(10,10))
plt.title('Comments Distribution')
plt.xlabel('Comments')
plt.hist(df['n_comments'].dropna(), bins = 30, label = 'Comments')
plt.legend()
plt.savefig("Comments Distribution.png", dpi=200)
plt.show()


# Line chart of number of AVG number of posts, upvotes, views, comments by year
VotesLine = df.groupby(df.votedon.dt.year)['votes'].agg('mean')
PostViewsLine = df.groupby(df.votedon.dt.year)['viewed'].agg('mean')
CommentsLine = df.groupby(df.votedon.dt.year)['n_comments'].agg('mean')

# We should be able to apply the .agg function
# to get an accruate count of posts in a year

PostsCountLine = df.groupby(df.votedon.dt.year)['etitle'].agg('count')


Years = PostsCountLine.reset_index()['votedon']

plt.figure(6, figsize=(10,10))
plt.plot(PostsCountLine, label='Number of Pictures Posted')
plt.plot(VotesLine, label='Average Up Votes')
plt.plot(PostViewsLine, label='Average Post Views')
plt.plot(CommentsLine, label='Average Comments Posted')
plt.legend()
plt.yscale('log')
plt.xlabel('')
plt.ylabel('')
plt.title('')
plt.savefig("Line Graph.png", dpi=200)
plt.show()


# distribution of independent variable (upvotes)
plt.figure(7, figsize=(10,10))
plt.ylabel('Frequency')
plt.title('Votes Distribution')
sns.distplot(df['votes'] )
plt.savefig("Votes Distribution.png", dpi=200)
plt.show()


## distribution of up votes is not normal. this can reduce preformance of ML models
## lets create votes_log and graph to see difference
plt.figure(8, figsize=(10,10))
plt.ylabel('Frequency')
plt.title('votes_log distribution')
sns.distplot(df['votes_log'] )
plt.savefig("votes_log distribution.png", dpi=200)
plt.show()


## check relationship of categorical variables to votes
categorical_feats = df.dtypes[df.dtypes == "object"].index


for catg in list(categorical_feats) :
    bp= df.boxplot(column=['votes_log'], by=[catg])
# medians look higher for recent years


## check relationship of numerical variables to votes
numerical_feats = ['viewed', 'n_comments', 'author_range',
                   'image_age', 'post_age'  ]
for num in list(numerical_feats) :
    sns.jointplot(num, 'votes_log', data = df, kind="hex")
    plt.show()


"""
Modeling: Create dummies for categorical features 

Split Data into Training/Test, Apply RF, Measure r^2
"""

cols_to_keep = ['etitle','viewed','n_comments','day_of_week'
                ,'MonthPosted','YearPosted','author_range',
                'image_age','post_age','votes_log']

ie = df[cols_to_keep]
ie = pd.get_dummies(ie)

predictors = ie.drop('votes_log', 1)
votes_log = ie['votes_log']


X_train, X_test, y_train, y_test = tts(predictors, votes_log, 
                                       test_size=0.3)
model = RandomForestRegressor()
model = model.fit(X_train, y_train)
print('Model Training and Testing Score ')
print(model.score(X_train, y_train), model.score(X_test, y_test))

filename = 'model.p'
pickle.dump(model, open(filename, 'wb'))


#Histogram of difference Between actual and Predicted Votes

plt.figure(9, figsize=(10,10))
plt.title('True and Predicted upvotes')
plt.xlabel('Actual - Predicted')
plt.hist(y_test - model.predict(X_test),bins = 20,label="Difference Between True and Predicted")
plt.grid()
plt.legend()
plt.savefig("True and Predicted upvotes.png", dpi=200)
plt.show()

'''
Cross Validation
'''
#Use KFold cross validation make sure no overfitting

results={}

cv = KFold(n_splits=5,shuffle=True,random_state=100)
r2 = make_scorer(r2_score)
r2_val_score = cross_val_score(RandomForestRegressor(), predictors, votes_log, cv=cv,scoring=r2)
scores= [r2_val_score.mean()]
results["RandomForest"]=scores
print('Average value of Kfold cross validation, K = 5')
print(results)

'''
Filter out Features 
'''

#1. Select Top 60% of features by f_regression

X_scored = SelectPercentile(score_func=f_regression, percentile=60).fit_transform(X_train, y_train)
                           
X_train, X_test, y_train, y_test = tts(X_scored, y_train, test_size=0.3)
model = RandomForestRegressor()
model.fit(X_train, y_train)
print('Score of model after filtering out 60% of features by f_regression')
print(model.score(X_train, y_train), model.score(X_test, y_test))


#2. Use variance threshold of 0.8 to reduce features

p = 0.2 # max allowed fraction of ones
sel = VarianceThreshold(threshold=(p * (1 - p)))
X_new = sel.fit_transform(predictors)

X_train, X_test, y_train, y_test = tts(X_new, votes_log, 
                                       test_size=0.3)
model = RandomForestRegressor()
model.fit(X_train, y_train)
print('Score of model after filtering out features with Variance Threshold of 0.8')
print(model.score(X_train, y_train), model.score(X_test, y_test))


