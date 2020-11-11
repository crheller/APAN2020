import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# make dummy data with strong linear relationship + noise
X = np.arange(0, 50, 0.1)[:, np.newaxis]
y = 3 * X + np.random.normal(0, 30, X.shape)

# plot
f, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(X, y)

# get cross validated rsquared, 20-fold CV
val_r2 = cross_val_score(LinearRegression(), X, y, scoring='r2', cv=10)

# get full-data rsquared
lr = LinearRegression()
lr.fit(X, y)
r2 = lr.score(X, y)

ax[1].hist(val_r2, label='val set r2', bins=np.arange(-0.5, 0.5, 0.02))
ax[1].axvline(r2, label='Full Dataset r2', lw=1, color='k')
ax[1].legend()

f.tight_layout()

plt.show()
