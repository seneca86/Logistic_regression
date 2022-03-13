# %% Odds
import pandas as pd
import numpy as np
p = np.arange(start=0.0, step=0.1, stop=1.1)
o = p / (1 - p + 1e-12)
p = o / (o + 1)
# %%
log_o = np.log(o)
o = np.exp(log_o)
p = o / (o + 1)
# %% Estimating parameters
y = np.array([0, 1, 0, 1])
x1 = np.array([0, 0, 0, 1])
x2 = np.array([0, 1, 1, 1])
# %%
beta = np.array([-1.5, 2.8, 1.1])
log_o = beta[0] + beta[1] * x1 + beta[2] * x2
print(f'{log_o=}')
# %%
o = np.exp(log_o)
p = o / (o + 1)
print(f'{p=}')
# %%
likelihood = np.prod(y * p + (1-y) * (1-p))
print(likelihood)
# %% Implementation
df = pd.read_csv('.lesson/assets/FemPreg.csv')
live = df.query('outcome == 1 & prglngth > 30') # filter live births
firsts = df.query('birthord == 1 & outcome == 1') # filter first borns
others = df.query('birthord != 1 & outcome == 1') # filter non-first borns
# %%
live = live.assign(boy = (live.babysex == 1).astype(int))
# %%
import statsmodels.formula.api as smf
model = smf.logit('boy ~ agepreg', data = live)
results = model.fit()
results.summary()
# %%
formula = 'boy ~ agepreg + hpagelb + birthord + C(race)'
model = smf.logit(formula, data=live)
results = model.fit()
results.summary()
# %%
actual = model.endog
baseline = actual.mean()
print(f'{baseline=}')
# %%
predict = (results.predict() >= 0.5)
true_pos = predict * actual
true_neg = (1 - predict) * (1 - actual)
acc = (sum(true_pos) + sum(true_neg)) / len(actual)
print(f'{acc=}')
# %%
new = pd.DataFrame([[35, 39, 3, 2]], columns=['agepreg', 'hpagelb', 'birthord', 'race'])
y = results.predict(new)
print(f'The chances of having a boy are {y}')
# %%
