# Demo 1 - Bayesian Lasso

n = 50
k = 100

np.random.seed(1234)
X = np.random.normal(size=(n, k))

beta = np.zeros(shape=k)
beta[[10,30,50,70]] =  10
beta[[20,40,60,80]] = -10

y = X @ beta + np.random.normal(size=n)


## Naive Model


with pm.Model() as bayes_lasso:
  b = pm.Laplace("beta", 0, 1, shape=k)#lam*tau, shape=k)
y_est = X @ b

s = pm.HalfNormal('sigma', sd=1)

likelihood = pm.Normal("y", mu=y_est, sigma=s, observed=y)

trace = pm.sample(return_inferencedata=True, random_seed=1234)


az.summary(trace)
az.summary(trace).iloc[[0,10,20,30,40,50,60,70,80,100]]


ax = az.plot_forest(trace)
plt.tight_layout()
plt.show()




## Plot helper

def plot_slope(trace, prior="beta", chain=0):
  post = (trace.posterior[prior]
          .to_dataframe()
          .reset_index()
          .query("chain == 0")
  )

sns.catplot(x="beta_dim_0", y="beta", data=post, kind="boxen", linewidth=0, color='blue', aspect=2, showfliers=False)
plt.tight_layout()
plt.show()



plot_slope(trace)

## Weakly Informative Prior


with pm.Model() as bayes_weak:
  b = pm.Normal("beta", 0, 10, shape=k)
y_est = X @ b

s = pm.HalfNormal('sigma', sd=2)

likelihood = pm.Normal("y", mu=y_est, sigma=s, observed=y)

trace = pm.sample(return_inferencedata=True, random_seed=12345)

ax = az.plot_forest(trace)
plt.tight_layout()
plt.show()

plot_slope(trace)

