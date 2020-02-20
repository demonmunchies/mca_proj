import mca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
pd.set_option('display.precision', 5)
pd.set_option('display.max_columns', 25)

data = pd.read_table('mod_nces_early_2016.csv', sep=',', skiprows=0, index_col=0, header=0)
data = data.clip(lower = 0)

metric_labels = data['TTLHHINC'].values

color_blind_labels = ['blue' if x < 3 else 'yellow' if x > 3 and x < 7 else 'red' for x in metric_labels]

X = data.drop('TTLHHINC', axis=1)


ncols = 94

mca_ben = mca.MCA(X, ncols=ncols)
mca_ind = mca.MCA(X, ncols=ncols, benzecri=False)

fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
table = pd.DataFrame(columns=X.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))

table.loc[fs,    :] = mca_ben.fs_r(N=2).T
table.loc[cos,   :] = mca_ben.cos_r(N=2).T
table.loc[cont,  :] = mca_ben.cont_r(N=2).T * 1000

factor_1 = mca_ben.fs_c(N=2).T[0]
factor_2 = mca_ben.fs_c(N=2).T[1]
col_labels = X.columns

factor_for_income = mca_ben.fs_r(N=2).T[1]

factor_1_sorted_values = [x for x in sorted(abs(factor_1), reverse=True)]
factor_2_sorted_values = [x for x in sorted(abs(factor_2), reverse=True)]

factor_1_sorted_labels = [x for _, x in sorted(zip(abs(factor_1), col_labels), reverse=True)]
factor_2_sorted_labels = [x for _, x in sorted(zip(abs(factor_2), col_labels), reverse=True)]

plt.figure()
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('Variable')
plt.ylabel('Magnitude in Factor 1')
plt.bar(factor_1_sorted_labels, factor_1_sorted_values)
plt.show()

plt.figure()
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('Variable')
plt.ylabel('Magnitude in Factor 2')
plt.bar(factor_2_sorted_labels, factor_2_sorted_values)
plt.show()



km = KMeans(n_clusters=4, n_init=10, max_iter=300)

k_means_labels = km.fit_predict(X)

points = table.loc[fs].values
labels = table.columns.values

plt.figure()
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# color blind mode
# plt.scatter(*points, s=120, marker='o', c=color_blind_labels, alpha=.5, linewidths=0)

# normal people mode
plt.scatter(*points, s=120, marker='o', c=k_means_labels, alpha=.5, linewidths=0)

plt.colorbar(label='K-Means Label (K = 4)');
plt.show()

plt.figure()
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('Income Class')
plt.ylabel('Factor 2')
plt.scatter(x = metric_labels, y = factor_for_income, marker = 'o')
plt.show()

for i in range(94):
	if factor_2_sorted_values[i] < .2:
		try:
			X = X.drop(factor_2_sorted_labels[i], axis=1)
			ncols = ncols - 1
		except:
			print('Oops! Something went horribly wrong!')

print(ncols)

print('Check 1')

mca_ben = mca.MCA(X, ncols=ncols)
mca_ind = mca.MCA(X, ncols=ncols, benzecri=False)

print('Check 2')

fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
table = pd.DataFrame(columns=X.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))

print('Check 3')

table.loc[fs,    :] = mca_ben.fs_r(N=2).T
# table.loc[cos,   :] = mca_ben.cos_r(N=2).T
# table.loc[cont,  :] = mca_ben.cont_r(N=2).T * 1000

print('Check 4')

points = table.loc[fs].values
labels = table.columns.values

plt.figure()
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

# color blind mode
# plt.scatter(*points, s=120, marker='o', c=color_blind_labels, alpha=.5, linewidths=0)

# normal people mode
plt.scatter(*points, s=120, marker='o', c=metric_labels, alpha=.5, linewidths=0)

plt.colorbar(label='Income Tier (Low to High)');
plt.show()
