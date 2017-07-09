#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from itertools import compress
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Combine financial and email features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', \
                'loan_advances', 'bonus', 'restricted_stock_deferred', \
                'deferred_income', 'total_stock_value', 'expenses', \
                'exercised_stock_options', 'other', 'long_term_incentive', \
                 'restricted_stock', 'director_fees', 'to_messages', \
                 'from_poi_to_this_person', 'from_messages', \
                 'from_this_person_to_poi', 'shared_receipt_with_poi']
# You will need to use more features
print "Number of features", len(features_list[1:])
### Load the dictionary containing the dataset
with open(path, "r") as data_file:
    data_dict = pickle.load(data_file)
print "Number of data points", len(data_dict)
### Task 2: Remove outliers
# Remove 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK' entries.
# Remove 'LOCKHART EUGENE E' since entry only has 'NaN' values
print data_dict['LOCKHART EUGENE E']
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['LOCKHART EUGENE E']
### Task 3: Create new feature(s)
# New feature combines salary, bonus and total_stock_value. Add new feature
# to data_dict.
new_feature = ['salary','bonus', 'total_stock_value']
feature_nan = defaultdict(int) # Count number of 'NaN' values
for name, info in data_dict.iteritems():
    total = 0
    for feat, val in info.iteritems():
        if val == 'NaN':
            feature_nan[feat] += 1
        if feat in new_feature and val != 'NaN':
            total += val
    data_dict[name]['total'] = total
# Print number of 'NaN' values for each feature
for key, value in feature_nan.iteritems():
    print key, value

### Store to my_dataset for easy export below.
# Add new feature to features_list.
my_dataset = data_dict
features_list.append('total')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Create dataframe of data for exploration
features_df = pd.DataFrame(features, columns = features_list[1:])
poi_df = pd.DataFrame(labels, columns = ['poi'])
features_poi_df = pd.concat([features_df, poi_df], axis = 1)
features_poi_df.describe()

#Count of all poi
print "Number of poi's:", features_poi_df.poi.sum()
print features_poi_df.count()

#Create pipeline and parameters with MinMaxScaler, SelectKBest,
#DecisionTreeClassifier
my_pipeline = make_pipeline(MinMaxScaler(), SelectKBest(), \
              DecisionTreeClassifier())
param_grid = dict(selectkbest__k = range(1,15), \
             decisiontreeclassifier__criterion = ['gini', 'entropy'], \
             decisiontreeclassifier__min_samples_split = range(2,20))

#Use StratifiedShuffleSplit for cross validation. Use GridSearchCV to find
#best classifier. Set scoring to f1.
sss= StratifiedShuffleSplit(n_splits= 100, test_size= 0.1, random_state= 42)
grid_search = GridSearchCV(my_pipeline, param_grid, scoring = 'f1', cv = sss)

#Fit to features and labels and return best estimator.
grid_search.fit(features, labels)
clf = grid_search.best_estimator_

# Make predictions on features. Find accuracy, precision and recall.
pred = clf.predict(features)
print accuracy_score(labels, pred)
print precision_score(labels, pred)
print recall_score(labels,pred)

# Obtain features selected from pipeline
support = grid_search.best_estimator_.named_steps['selectkbest'].get_support()
selected_features = list(compress(features_list[1:], support))
print selected_features

# Obtain feature scores.
scores = grid_search.best_estimator_.named_steps['selectkbest'].scores_
selected_features_scores = list(compress(scores, support))
print selected_features_scores

# Graph selected features for further outlier exploration.


matplotlib.style.use('ggplot')

#Use pandas dataframe for graphing
features_financial = features_poi_df.loc[0:,'salary':'director_fees']
features_financial = pd.concat([features_financial, poi_df], axis = 1)
features_financial = pd.concat((features_financial, \
                     features_poi_df[ 'total']), axis = 1, join = 'inner')

# Graph distributions of features
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(4,18))

for i, ax in enumerate(axes.flatten()):

    sns.distplot(features_financial.loc[:, selected_features[i]], \
    kde= False,ax=ax)



dump_classifier_and_data(clf, my_dataset, features_list)
