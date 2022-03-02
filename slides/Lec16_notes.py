## Setup

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


from sklearn.linear_model import LogisticRegression

email = pd.read_csv(
  'https://sta663-sp22.github.io/slides/data/email.csv'
).loc(
  ['spam', 'exclaim_mess', 'format', 'num_char', 'line_breaks', 'number']
)

email_dc = pd.get_dummies(email)
email_dc


y = email_dc.spam
X = email_dc.drop('spam', axis=1)

m = LogisticRegression(fit_intercept = False).fit(X, y)



## DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

tree_gs = GridSearchCV(
  DecisionTreeClassifier(),
  param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2,3,4,5,6,7]
  },
  cv = KFold(10, shuffle=True, random_state=1234),
  scoring = "roc_auc",
  n_jobs = 4
).fit(
  X, y
)

tree_gs.best_estimator_
tree_gs.best_score_

for p, s in  zip(tree_gs.cv_results_["params"], tree_gs.cv_results_["mean_test_score"]):
  print(p,"Score:",s)

from sklearn.metrics import classification_report, confusion_matrix

confusion_matrix(y, tree_gs.best_estimator_.predict(X))

print(
  classification_report(y, tree_gs.best_estimator_.predict(X))
)


from sklearn.metrics import auc, roc_curve, RocCurveDisplay

fpr, tpr, thresholds = roc_curve(y, tree_gs.best_estimator_.predict_proba(X)[:,1])
roc_auc = auc(fpr, tpr)

RocCurveDisplay(
  fpr=fpr, tpr=tpr, roc_auc=roc_auc,
  estimator_name='Tree Classifier'
).plot()

plt.show()


## SVC

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

svc_pipe = make_pipeline(
  StandardScaler(),
  SVC()
)

svc_gs = GridSearchCV(
  svc_pipe,
  param_grid = [
    {"svc__kernel": ["rbf"], "svc__C": [1, 10, 100, 1000]},
    {"svc__kernel": ["linear"], "svc__C": [1, 10, 100, 1000]},
    {"svc__kernel": ["poly"], "svc__C": [1, 10, 100, 1000], "svc__degree": [2,3,4,5]},
  ],
  cv = KFold(5, shuffle=True, random_state=1234),
  scoring = "roc_auc",
  n_jobs = 6
).fit(
  X, y
)

svc_gs.best_estimator_
svc_gs.best_score_

for p, s in  zip(svc_gs.cv_results_["params"], svc_gs.cv_results_["mean_test_score"]):
  print(p,"Score:",s)

print(
  classification_report(y, svc_gs.best_estimator_.predict(X))
)


## Digits + Classification Tree


from sklearn.datasets import load_digits
digits = load_digits(as_frame=True)


X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=1234
)

digits_tree = GridSearchCV(
  DecisionTreeClassifier(),
  param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": range(2,16)
  },
  cv = KFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(
  X_train, y_train
)

digits_tree.best_estimator_
digits_tree.best_score_

accuracy_score(y_test, digits_tree.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, digits_tree.best_estimator_.predict(X_test)
)


### GridSearchCV w/ multiple models

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

p = Pipeline([
  ("model", DecisionTreeClassifier())
])


digits_tree = GridSearchCV(
  p,
  param_grid = {
    "model": [
      DecisionTreeClassifier(),
      RandomForestClassifier()
    ],
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": range(2,10)
  },
  cv = KFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(
  X_train, y_train
)

digits_tree.best_estimator_
digits_tree.best_score_

accuracy_score(y_test, digits_tree.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, digits_tree.best_estimator_.predict(X_test)
)
