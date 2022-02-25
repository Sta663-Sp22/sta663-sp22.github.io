import numpy as np
import pandas as pd
import seaborn as sns

## Demo 1


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline, make_union
from sklearn.compose import make_column_selector, make_column_transformer

books = pd.read_csv("https://sta663-sp22.github.io/slides/data/daag_books.csv")

p = make_pipeline(
  make_column_transformer(
    (OneHotEncoder(drop="first"), make_column_selector(dtype_include=object)),
    remainder = "passthrough"
  ),
  PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
  LinearRegression()
)

#p.fit(X = books.drop(["weight"], axis=1))
#p.transform(books.drop(["weight"], axis=1))
#p.get_feature_names_out()

p.fit(
  X = books.drop(["weight"], axis=1),
  y = books.weight
)

p.named_steps["linearregression"].intercept_
p.named_steps["linearregression"].coef_
p.get_feature_names_out()
p[:-1].get_feature_names_out()
p.get_params().keys()


## Exercise 1


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold

from sklearn import datasets
X, y = datasets.load_diabetes(return_X_y=True)

p = make_pipeline(
    StandardScaler(),
    Lasso()
)

gs = GridSearchCV(
  p,
  param_grid = {"lasso__alpha": np.logspace(-4, 1, 100)},
  scoring = 'neg_root_mean_squared_error',
  cv = KFold(10, shuffle=True, random_state=12345)
).fit(
  X, y
)

gs.best_params_
gs.best_index_
gs.best_score_
gs.best_estimator_


## Uncertainty

alpha = np.array(gs.cv_results_["param_lasso__alpha"], dtype="float64")
score = -gs.cv_results_["mean_test_score"]
score_std = gs.cv_results_["std_test_score"]
n_folds = gs.cv.get_n_splits()

plt.figure(layout="constrained")

ax = sns.lineplot(x=alpha, y=score)
ax.set_xscale("log")

plt.fill_between(
  x = alpha,
  y1 = score + 1.96*score_std / np.sqrt(n_folds),
  y2 = score - 1.96*score_std / np.sqrt(n_folds),
  alpha = 0.2
)

ax.set_xlim(1e-5, 1)
ax.set_ylim(54.4, 54.6)

plt.show()



### Traceplot

alpha = np.logspace(-4, 2, 100)
betas = []

for a in alpha:
    p = p.set_params(lasso__alpha = a)
    p = p.fit(X, y)
    
    betas.append(p.named_steps["lasso"].coef_)

res = pd.DataFrame(
  data = betas, columns = p[:-1].get_feature_names_out()
).assign(
  alpha = alpha  
)

res


g = sns.relplot(
  data = res.melt(id_vars="alpha", value_name="coef values", var_name="feature"),
  x = "alpha", y = "coef values", hue = "feature",
  kind = "line", aspect=2
)
g.set(xscale="log")
plt.axvline(x = gs.best_params_["lasso__alpha"], color="k", linestyle="--")
plt.show()
