from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import check_X_y, check_is_fitted



class interact_features(BaseEstimator, TransformerMixin):
  def __init__(self, interaction_only = False, include_intercept = False):
    self.interaction_only = interaction_only
    self.include_intercept = include_intercept
  
  def fit(self, X, y=None):
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L495
    self._validate_data(X=X, reset=True, ensure_min_features=2)
    self._check_feature_names(X=X, reset=True)

    return self
  
  def transform(self, X, y=None):
    check_is_fitted(self, "n_features_in_")
    
    self._validate_data(X=X, reset=False)
    self._check_feature_names(X=X, reset=False)
    
    X = np.array(X)
    
    new_cols = []
    for i in range(self.n_features_in_-1):
      for j in range(1, self.n_features_in_):
        new_cols.append( X[:,i] * X[:,j] )

    new_X = np.column_stack(new_cols)
    
    if not self.interaction_only:
      new_X = np.column_stack([X, new_X])

    if self.include_intercept:
      new_X = np.column_stack([np.ones((new_X.shape[0],1)), new_X])

    
    return new_X

  def get_feature_names_out(self):
    check_is_fitted(self, "n_features_in_")
    
    if not hasattr(self, "feature_names_in_"):
      feat_names = ["x"+str(i) for i in range(self.n_features_in_)]
    else:
      feat_names = self.feature_names_in_
    
    new_feat_names = []
    for i in range(self.n_features_in_-1):
      for j in range(1, self.n_features_in_):
        new_feat_names.append( feat_names[i] + " * " + feat_names[j] )
    
    if not self.interaction_only:
      new_feat_names = feat_names + new_feat_names
      
    if self.include_intercept:
      new_feat_names = ["1"] + new_feat_names
    
    return new_feat_names


X = pd.DataFrame({"x1": range(1,6), "x2": range(5, 0, -1)})
Y = pd.DataFrame({"x1": range(1,6)})
Z = np.array(X)

itf = interact_features().fit(X)
itf.n_features_in_
itf.feature_names_in_
itf.transform(X)

itf2 = interact_features().fit(Z)
itf2.n_features_in_
itf2.feature_names_in_
itf2.transform(Z)
itf2
itf2.get_feature_names_out()

