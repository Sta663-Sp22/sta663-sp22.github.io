import matplotlib.pyplot as plt

pts = np.linspace(-1,3, 5000)
x, y = np.meshgrid(pts, pts)

f = (1-x)**2 + 100*(y-x**2)**2


np.min(f)
x[f == np.min(f)]
y[f == np.min(f)]

min_i = np.argmin(f, axis=None)
x.reshape(-1)[min_i]
y.reshape(-1)[min_i]


##

rng = np.random.default_rng(1234)

d = rng.normal(loc=[-1,0,1], scale=[1,2,3], size=(1000,3))
d.mean(axis=0)
d.std(axis=0)

ds = (d - d.mean(axis=0)) / d.std(axis=0)
ds.mean(0)
ds.std(0)
