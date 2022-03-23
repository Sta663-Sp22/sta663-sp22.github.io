# Demo 1 - Linear regression

Given the below data, we will fit a linear regression model to the following synthetic data,

np.random.seed(1234)
n = 11
m = 6
b = 2
x = np.linspace(0, 1, n)
y = m*x + b + np.random.randn(n)

## Model

with pm.Model() as lm:
  m = pm.Normal('m', mu=0, sd=50)
  b = pm.Normal('b', mu=0, sd=50)
  sigma = pm.HalfNormal('sigma', sd=5)
  
  likelihood = pm.Normal('y', mu=m*x + b, sd=sigma, observed=y)
  
  trace = pm.sample(return_inferencedata=True, random_seed=1234)

az.summary(trace)

ax = az.plot_trace(trace)
plt.show()

ax = az.plot_posterior(trace, ref_val=[6,2,1])
plt.show()


## Posterior Predictive 1

plt.scatter(x, y, s=30, label='data')

post_m = trace.posterior['m'][0, -500:]
post_b = trace.posterior['b'][0, -500:]

plt.figure(layout="constrained")
plt.scatter(x, y, s=30, label='data')
for m, b in zip(post_m.values, post_b.values):
    plt.plot(x, m*x + b, c='gray', alpha=0.1)
plt.plot(x, 6*x + 2, label='true regression line', lw=3., c='red')
plt.legend(loc='best')
plt.show()


## Posterior Predictive 2


with lm:
  pp = pm.sample_posterior_predictive(trace, samples=200)
  
pp['y'].shape


plt.figure(layout="constrained")
plt.plot(x, pp['y'].T, c="grey", alpha=0.1)
plt.scatter(x, y, s=30, label='data')
plt.show()


## Model revision


with pm.Model() as lm2:
  m = pm.Normal('m', mu=0, sd=50)
  b = pm.Normal('b', mu=0, sd=50)
  sigma = pm.HalfNormal('sigma', sd=5)
  
  y_est = pm.Deterministic("y_est", m*x + b)
  
  likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
  
  trace = pm.sample(return_inferencedata=True, random_seed=1234)
  pp = pm.sample_posterior_predictive(trace, var_names=["y_est"], samples=200)


## Posterior Predictive 3

plt.figure(layout="constrained")
ax = az.plot_trace(trace, compact=False, figsize=(6,12))
plt.show()


pp['y_est'].shape

plt.figure(layout="constrained")
plt.plot(x, pp['y_est'].T, c="grey", alpha=0.1)
plt.scatter(x, y, s=30, label='data')
plt.show()


