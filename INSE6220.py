import numpy as np
import pandas as pd

import matplotlib.pyplot as plt;
from matplotlib.ticker import FixedLocator

# use seaborn plotting style defaults
import seaborn as sns
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (12,8)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from scipy.stats import beta
from scipy.stats import f

from itertools import cycle

# Machine Learning Modules
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from bioinfokit.visuz import cluster

# from google.colab import files

# uploaded = files.upload()

#read cvs file into dataframe
df = pd.read_csv('https://raw.githubusercontent.com/ZaraJ2023/Concordia-INSE6220-F2023/main/Raisin_Dataset01.csv')
df.head(n=25)
df.info()
y = df['class']
target = df['class'].to_numpy()
print("Number of duplicated rows is: ", df.duplicated().sum())

print("Number of rows with NaNs is: ", df.isna().any(axis=1).sum())
"""
sns.pairplot(df, hue='class')
plt.show()

y = df['class']
y.value_counts().plot(kind='pie')
plt.ylabel('')
plt.show()



X = df.iloc[:,0:7]
X.head(10)

X.describe().transpose()

X_st = StandardScaler().fit_transform(X)
df = pd.DataFrame(X_st)
df.columns = X.columns
df.describe().transpose()

observations = list(df.index)
print(observations)
variables = list(df.columns)
print(variables)

y.value_counts().plot(kind='bar', rot=0, color='orange')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

ax = plt.figure()
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

# # Use swarmplot() or stripplot to show the datapoints on top of the boxes:
# #plt. figure()
ax = plt.figure()
ax = sns.boxplot(data=df, orient="v", palette="Set2")
ax = sns.stripplot(data=df, color="navy")
ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()

# #5number summary
df.describe()
plt.show()


# #pair plot
sns.pairplot(df)
plt.show()


# ## **Covariance**
dfc = df - df.mean() #centered data
ax = sns.heatmap(dfc.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True,
            cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45);
plt.title('Covariance matrix')
plt.show()

# ### **Principal Component Analysis (PCA)**
pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)
#
# 1(Kecimen)->0,  2(Besni)->1
idx_Kecimen = np.where(y == 0)
idx_Besni = np.where(y == 1)
#
plt. figure()
plt.scatter(Z[idx_Kecimen,0], Z[idx_Kecimen,1], c='r', label='Kecimen (0)')
plt.scatter(Z[idx_Besni,0], Z[idx_Besni,1], c='g', label='Besni (1)')
plt.legend()
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
#for label, x, y in zip(observations,Z[:, 0],Z[:, 1]):
#    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
#        textcoords='offset points', ha='right', va='bottom')
plt.show()

### **Eigenvectors**
A = pca.components_.T
#print(f'Eigenvector matrix:\n{A}')

plt.figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$')
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
  plt.annotate(label, xy=(x, y), xytext=(-2, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

plt.figure()
plt.scatter(A[:, 0],A[:, 1], marker='o', c=A[:, 2], s=A[:, 3]*500, cmap=plt.get_cmap('Spectral'))
plt.xlabel('$A_1$')
plt.ylabel('$A_2$')
for label, x, y in zip(variables,A[:, 0],A[:, 1]):
  plt.annotate(label,xy=(x, y), xytext=(-20, 20),
      textcoords='offset points', ha='right', va='bottom',
      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
      arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()

### **Scree plot**
#Eigenvalues
Lambda = pca.explained_variance_
#print(f'Eigenvalues:\n{Lambda}')

#Scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/sum(Lambda), 'ro-', lw=3)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

### **Explained Variance**
ell = pca.explained_variance_ratio_
plt.figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

##**Explained Variance per PC**
PC_variance = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
PC_variance

###### **Biplot**
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[:,0]
A2 = A[:,1]
Z1 = Z[:,0]
Z2 = Z[:,1]

plt.figure()
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
for i in range(len(A1)):
# arrows project features as vectors onto PC axes
  plt.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2), color='orange', width=0.0005, head_width=0.0025)
  plt.text(A1[i]*max(Z1)*1.2, A2[i]*max(Z2)*1.2,variables[i], color='k')

plt.scatter(Z[idx_Kecimen,0], Z[idx_Kecimen,1], c='r', label='Kecimen (0)')
plt.scatter(Z[idx_Besni,0], Z[idx_Besni,1], c='g', label='Besni (1)')
plt.legend(loc='upper left')
plt.show()
#for i in range(len(Z1)):
# circles project documents (ie rows from csv) as points onto PC axes
  #plt.scatter(Z1[i], Z2[i], c='g', marker='o')
  #plt.text(Z1[i]*1.2, Z2[i]*1.2, observations[i], color='b')
#cluster.biplot(cscore=Z, loadings=A, labels=X.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
#               var2=round(pca.explained_variance_ratio_[1]*100, 2), colorlist=target)

#**Using PCA Librarry**
#!pip install pca

from pca import pca
# Initialize and keep all PCs
model = pca()
# Fit transform
out = model.fit_transform(df)
 # Print the top features. The results show that f1 is best, followed by f2 etc
print(out['topfeat'])

# Plot PC1 vs PC2
model.plot();
plt.show()

# Create a biplot without label and legend
ax = model.biplot(legend=False)
plt.show()


# Create a biplot with a specific colormap, without label and legend

model.biplot(cmap='RdYlGn_r', label=False, legend=False)
plt.show()
# Create a 3D biplot without legend
ax = model.biplot3d(legend=False)

plt.show()
##### **Principal components**

comps = pd.DataFrame(A, columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True,
            cbar=True, square=True)
ax = sns.heatmap(comps, cmap='RdYlGn_r', linewidths=0.5, annot=True, cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.xticks(rotation=90)
plt.title('Principal components')
plt.show()

print(f'PC1:{A1}')
print(f'PC2:{A2}')



##Hotelling's T2 test
alpha = 0.05
p=Z.shape[1]
n=Z.shape[0]

UCL=((n-1)**2/n )*beta.ppf(1-alpha, p / 2 , (n-p-1)/ 2)
UCL2=p*(n+1)*(n-1)/(n*(n-p) )*f.ppf(1-alpha, p , n-p)
Tsquare=np.array([0]*Z.shape[0])
for i in range(Z.shape[0]):
  Tsquare[i] = np.matmul(np.matmul(np.transpose(Z[i]),np.diag(1/Lambda) ) , Z[i])

fig, ax = plt.subplots()
ax.plot(Tsquare,'-b', marker='o', mec='y',mfc='r' )
ax.plot([UCL for i in range(len(Z1))], "--g", label="UCL")
plt.ylabel('Hotelling $T^2$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))
plt.show()

print(np.argwhere(Tsquare>UCL))



####### **Control Charts for Principal Components**
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(Z1,'-b', marker='o', mec='y',mfc='r')
ax1.plot([3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--g", label="UCL")
ax1.plot([-3*np.sqrt(Lambda[0]) for i in range(len(Z1))], "--b", label='LCL')
ax1.plot([0 for i in range(len(Z1))], "-", color='black',label='CL')
ax1.set_ylabel('$Z_1$')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))

ax2.plot(Z2,'-b', marker='o', mec='y',mfc='r')
ax2.plot([3*np.sqrt(Lambda[1]) for i in range(len(Z2))], "--g", label="UCL")
ax2.plot([-3*np.sqrt(Lambda[1]) for i in range(len(Z2))], "--b", label='LCL')
ax2.plot([0 for i in range(len(Z2))], "-", color='black',label='CL')
ax2.set_ylabel('$Z_2$')
ax2.set_xlabel('Sample Number')
legend = ax.legend(shadow=False, ncol=4, bbox_to_anchor=(0.85, -0.1))
plt.legend()

##### Out of Control Points
print(np.argwhere(Z1<-3*np.sqrt(Lambda[0])))
print(np.argwhere(Z1>3*np.sqrt(Lambda[0])))
print(np.argwhere(Z2<-3*np.sqrt(Lambda[1])))
print(np.argwhere(Z2>3*np.sqrt(Lambda[1])))

##############

"""
# check installed version
import pycaret
pycaret.__version__

def enable_colab():
 # Import the PyCaret library
 import pycaret
 pycaret.utils.enable_colab()
 from enable_colab import enable_colab
 enable_colab()

data = df.sample(frac=0.9, random_state=786)
data_unseen = df.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

from pycaret.classification import *
clf = setup(data=data, target='class', train_size=0.7, session_id=123)


# loading sample dataset from pycaret dataset module
from pycaret.datasets import get_data
data = get_data('diabetes')

# import pycaret classification and init setup
from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)


# import ClassificationExperiment and init the class
from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()


# check the type of exp
type(exp)

# init setup on exp
exp.setup(data, target = 'Class variable', session_id = 123)

# compare baseline models
best = compare_models()

# compare models using OOP
exp.compare_models()

# plot confusion matrix
# plot_model(best, plot = 'confusion_matrix')

# plot AUC
plot_model(best, plot = 'auc')

# plot feature importance
plot_model(best, plot = 'feature')

evaluate_model(best)

# predict on test set
holdout_pred = predict_model(best)

# show predictions df
holdout_pred.head()

# copy data and drop Class variable

new_data = data.copy()
new_data.drop('Class variable', axis=1, inplace=True)
new_data.head()

# predict model on new_data
predictions = predict_model(best, data = new_data)
predictions.head()

###
# save pipeline
save_model(best, 'my_first_pipeline')

# load pipeline
loaded_best_pipeline = load_model('my_first_pipeline')
loaded_best_pipeline

#### Detailed function-by-function overview
# ✅ Setup
# init setup function
s = setup(data, target = 'Class variable', session_id = 123)

# check all available config
get_config()

# lets access X_train_transformed
get_config('X_train_transformed')

# another example: let's access seed
print("The current seed is: {}".format(get_config('seed')))

# now lets change it using set_config
set_config('seed', 786)
print("The new seed is: {}".format(get_config('seed')))

# help(setup)

# init setup with normalize = True

s = setup(data, target = 'Class variable', session_id = 123,
          normalize = True, normalize_method = 'minmax')

# lets check the X_train_transformed to see effect of params passed
get_config('X_train_transformed')['Number of times pregnant'].hist()

get_config('X_train')['Number of times pregnant'].hist()


###✅ Compare Models

best = compare_models()

# check available models
models()

compare_tree_models = compare_models(include = ['dt', 'rf', 'et', 'gbc', 'xgboost', 'lightgbm', 'catboost'])

compare_tree_models

compare_tree_models_results = pull()
compare_tree_models_results

best_recall_models_top3 = compare_models(sort = 'Recall', n_select = 3)

# list of top 3 models by Recall
best_recall_models_top3

# help(compare_models)

##✅ Set Custom Metrics

# check available metrics used in CV
get_metrics()

# create a custom function
import numpy as np

def custom_metric(y, y_pred):
    tp = np.where((y_pred==1) & (y==1), (100), 0)
    fp = np.where((y_pred==1) & (y==0), -5, 0)
    return np.sum([tp,fp])

# add metric to PyCaret
add_metric('custom_metric', 'Custom Metric', custom_metric)

# now let's run compare_models again
compare_models()

# remove custom metric
remove_metric('custom_metric')

##✅ Experiment Logging

# from pycaret.classification import *
# s = setup(data, target = 'Class variable', log_experiment='mlflow', experiment_name='diabetes_experiment')

# compare models
# best = compare_models()

# start mlflow server on localhost:5000
# !mlflow ui

# help(setup)


###✅ Create Model

# check all the available models
models()

# train logistic regression with default fold=10
lr = create_model('lr')

lr_results = pull()
print(type(lr_results))
lr_results

# train logistic regression with fold=3
lr = create_model('lr', fold=3)

# train logistic regression with specific model parameters
create_model('lr', C = 0.5, l1_ratio = 0.15)

# train lr and return train score as well alongwith CV
create_model('lr', return_train_score=True)

# change the probability threshold of classifier from 0.5 to 0.66
create_model('lr', probability_threshold = 0.66)

# help(create_model)


###✅ Tune Model

# train a dt model with default params
dt = create_model('dt')

# tune hyperparameters of dt
tuned_dt = tune_model(dt)

dt

# define tuning grid
dt_grid = {'max_depth' : [None, 2, 4, 6, 8, 10, 12]}

# tune model with custom grid and metric = F1
tuned_dt = tune_model(dt, custom_grid = dt_grid, optimize = 'F1')

# to access the tuner object you can set return_tuner = True
tuned_dt, tuner = tune_model(dt, return_tuner=True)

# model object
tuned_dt

# tuner object
tuner

# tune dt using optuna
tuned_dt = tune_model(dt, search_library = 'optuna')

# help(tune_model)

####✅ Ensemble Model

# ensemble with bagging
ensemble_model(dt, method = 'Bagging')

# ensemble with boosting
ensemble_model(dt, method = 'Boosting')

# help(ensemble_model)


#####✅ Blend Models

# top 3 models based on recall
best_recall_models_top3

# blend top 3 models
blend_models(best_recall_models_top3)

# help(blend_models)


###### **Multi-Class Classification**


# Test-Train Split
# Y = df['Perimeter']
#
# # Assuming the target variable is dropped from features
#
# X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=0)
# print(f'Train Dataset Size: {X_train.shape[0]}')
# print(f'Test Dataset Size: {X_test.shape[0]}')
#
# Z_train, Z_test, zy_train, zy_test = train_test_split(Z, Y, test_size=0.2, random_state=0)
# Z12_train, Z12_test, z12y_train, z12y_test = train_test_split(Z[:,:2], Y, test_size=0.2, random_state=0)
#
# # Define the evaluation metric
# scoring = ['f1_macro']

#Gaussian Naive Bayes (GNB)
# gnb = GaussianNB()
#
# datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test),
#             ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
# for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
#     gnb.fit(Xtr, ytr)
#     y_pred = gnb.predict(Xtst)
#     gnb_score = gnb.score(Xtst, ytst)
#
#     # Classification Report
#     print(f'DATASET: {name}')
#     print('Classification Report:')
#     print(classification_report(ytst, y_pred, digits=3))
#
#     # Confusion Matrix
#     cm_gnb = confusion_matrix(y_true=ytst, y_pred=y_pred)
#     ax = sns.heatmap(cm_gnb, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
#     plt.yticks(rotation=0)
#     ax.tick_params(labelbottom=False, labeltop=True)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#     # plt.title('Naive Bayes Confusion Matrix')
#     plt.show()
#
#     # ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
#     if name == 'Z12':
#         # Plotting decision regions
#         x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
#         y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#
#         Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
#
#         plt.contourf(xx, yy, Z, alpha=0.4)
#         plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
#         plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
#         plt.xlabel('$Z_1$')
#         plt.ylabel('$Z_2$')
#         plt.legend()
#         plt.show()
#
#     print(np.where(ytst != y_pred))


#####K Nearest Neighbors (KNN)
# # Hyperparameter grid search for k
# param_grid = {'n_neighbors': [2, 4, 8, 16, 32]}
# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn, param_grid, cv=5)
#
# # Find best k
# knn_full_data = grid_search.fit(X_train, y_train)
# knn_Z = grid_search.fit(Z_train, zy_train)
# knn_Z12 = grid_search.fit(Z12_train, z12y_train)
#
# # Get best k
# print('Grid Search Results:')
# k_full_data = knn_full_data.best_params_
# k_Z = knn_Z.best_params_
# k_Z12 = knn_Z12.best_params_
# print(f'k_full_data: {k_full_data}\nk_Z: {k_Z}\nk_Z12: {k_Z12}')
#
# # Apply best k
# knn = KNeighborsClassifier(n_neighbors=k_full_data.get('n_neighbors'))
# scores_knn_full_data = cross_validate(knn, X_train, y_train, cv=5, scoring=scoring)
# scores_knn_Z = cross_validate(knn, Z_train, zy_train, cv=5, scoring=scoring)
# scores_knn_Z12 = cross_validate(knn, Z12_train, z12y_train, cv=5, scoring=scoring)
#
# knn_scores_dict = {}
# for i in ['fit_time', 'test_f1_macro']:
#     knn_scores_dict["knn_full_data " + i] = scores_knn_full_data[i]
#     knn_scores_dict["knn_Z  " + i] = scores_knn_Z[i]
#     knn_scores_dict["knn_Z12 " + i] = scores_knn_Z12[i]
#
# knn_scores_data = pd.DataFrame(knn_scores_dict).T
# # knn_scores_data['avgs'] = knn_scores_data.mean(axis=1)
# print(f'{knn_scores_data}\n')
#
# datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test),
#             ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
# for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
#     # Apply on train-test split
#     knn.fit(Xtr, ytr)
#     y_pred = knn.predict(Xtst)
#     knn_score = knn.score(Xtst, ytst)
#     # print(f'\nTest Set Accuracy: {knn_score:.3f}')
#
#     # Classification Report
#     print(f'DATASET: {name}')
#     print('Classification Report:')
#     print(classification_report(ytst, y_pred, digits=3))
#
#     # Confusion Matrix
#     cm_knn = confusion_matrix(y_true=ytst, y_pred=y_pred)
#     ax = sns.heatmap(cm_knn, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
#     plt.yticks(rotation=0)
#     ax.tick_params(labelbottom=False, labeltop=True)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#     # plt.title('KNN Confusion Matrix')
#     plt.show()
#
#     # ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
#     if name == 'Z12':
#         # Plotting decision regions
#         x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
#         y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#
#         Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)
#
#         plt.contourf(xx, yy, Z, alpha=0.4)
#         plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
#         plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
#         plt.xlabel('$Z_1$')
#         plt.ylabel('$Z_2$')
#         plt.legend()
#         plt.show()
#
#         # print(np.where(ytst != y_pred))
#
#
# ######### Decision Trees (DT)
# # Hyperparameter search for DT depth
# param_grid = {'max_depth': [2, 4, 8, 16, 32, 64]}
# dt = DecisionTreeClassifier(random_state=0)
# grid_search = GridSearchCV(dt, param_grid, cv=5)
#
# # Find best depth
# dt_full_data = grid_search.fit(X_train, y_train)
# dt_Z = grid_search.fit(Z_train, zy_train)
# dt_Z12 = grid_search.fit(Z12_train, z12y_train)
#
# # Get best tree depth
# print('Grid Search Results:')
# depth_full_data = dt_full_data.best_params_
# depth_Z = dt_Z.best_params_
# depth_Z12 = dt_Z12.best_params_
# print(f'depth_full_data: {depth_full_data}\ndepth_Z: {depth_Z}\ndepth_Z12: {depth_Z12}')
#
# # Apply best k
# dt = DecisionTreeClassifier(max_depth=depth_full_data.get('max_depth'))
# scores_dt_full_data = cross_validate(dt, X_train, y_train, cv=5, scoring=scoring)
# scores_dt_Z = cross_validate(dt, Z_train, zy_train, cv=5, scoring=scoring)
# scores_dt_Z12 = cross_validate(dt, Z12_train, z12y_train, cv=5, scoring=scoring)
#
# dt_scores_dict={}
# for i in ['fit_time', 'test_f1_macro']:
#   dt_scores_dict["dt_full_data " + i ] = scores_dt_full_data[i]
#   dt_scores_dict["dt_Z  " + i ] = scores_dt_Z[i]
#   dt_scores_dict["dt_Z12 " + i ] = scores_dt_Z12[i]
#
# dt_scores_data = pd.DataFrame(dt_scores_dict).T
# #dt_scores_data['avgs'] = dt_scores_data.mean(axis=1)
# print(f'{dt_scores_data}\n')
#
# datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test), ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
# for i, (name, Xtr, ytr, Xtst, ytst) in enumerate(datasets):
#   # Apply on train-test split
#   dt.fit(Xtr, ytr)
#   y_pred = dt.predict(Xtst)
#   dt_score = dt.score(Xtst, ytst)
#   #print(dt_score)
#
#   # Classification Report
#   print(f'DATASET: {name}')
#   print('Classification Report:')
#   print(classification_report(ytst, y_pred, digits=3))
#
#   # Confusion Matrix
#   cm_dt = confusion_matrix(y_true=ytst, y_pred=y_pred)
#   ax = sns.heatmap(cm_dt, cmap='RdYlGn_r', linewidths=0.5, annot=True, square=True)
#   plt.yticks(rotation=0)
#   ax.tick_params(labelbottom=False,labeltop=True)
#   ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
#   #plt.title('Decision Tree Confusion Matrix')
#   plt.show()
#
#   #ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
#   if name == 'Z12':
#     # Plotting decision regions
#     x_min, x_max = Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1
#     y_min, y_max = Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#
#     Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     plt.contourf(xx, yy, Z, alpha=0.4)
#     plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, s=20, edgecolor="k", label='Train Set')
#     plt.scatter(Xtst[:, 0], Xtst[:, 1], c=ytst, marker='^', s=20, edgecolor="k", label='Test Set')
#     plt.xlabel('$Z_1$')
#     plt.ylabel('$Z_2$')
#     plt.legend()
#     plt.show()
#
#
# ##### **ROC Curves**
# #ADAPTED FROM: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#
# datasets = [('FULL DATA', X_train, y_train, X_test, y_test), ('Z', Z_train, zy_train, Z_test, zy_test), ('Z12', Z12_train, z12y_train, Z12_test, z12y_test)]
# for i, (name, X_tr, y_tr, X_tst, y_tst) in enumerate(datasets):
#   # Binarize the labels
#   y_train = label_binarize(y_tr, classes=[0, 1, 2])
#   y_test = label_binarize(y_tst, classes=[0, 1, 2])
#   n_classes = y_train.shape[1]
#   print(f'DATASET: {name}')
#
#   list_algos = [gnb, knn, dt]
#   algo_name = ['Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree']
#   for i, (algo, algo_name) in enumerate(zip(list_algos, algo_name)):
#     classifier = OneVsRestClassifier(algo)
#     y_pred = classifier.fit(X_tr, y_train).predict(X_tst)
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i], drop_intermediate=False)
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel(), drop_intermediate=False)
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#     # Plot all ROC curves
#     fig, ax = plt.subplots()
#
#     plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})', color="deeppink", linestyle=':')
#     plt.plot(fpr["macro"], tpr["macro"], label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', color="navy", linestyle=':')
#
#     colors = cycle(['c', 'm', 'r'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i],tpr[i], color=color,label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
#
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.xlim([-0.1, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title(f'{algo_name}')
#     plt.legend()
#     plt.show()
#
#
# #### **Bar Chart Plot**
# # ADAPTED FROM: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
# def autolabel(rects):
#     for r in rects:
#         height = r.get_height()
#         ax.annotate(f'{height}', xy=(r.get_x() + r.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
#
# n_groups = 3
# ind = np.arange(n_groups)
#
# # F1 Scores from above
# NB = (0.897, 0.955, 0.941)
# KNN = (0.906, 0.906, 0.903)
# DT = (0.918, 0.898, 0.870)
#
# # create plot
# fig, ax = plt.subplots(figsize=(10,7))
# index = np.arange(n_groups)
# bar_width = 0.20
# opacity = 0.8
#
# rects1 = plt.bar(index, NB, bar_width, alpha=opacity, color='b', label='Naive Bayes')
# rects2 = plt.bar(index + bar_width, KNN, bar_width, alpha=opacity, color='y', label='K-Nearest Neighbors')
# rects3 = plt.bar(index + bar_width*2, DT, bar_width, alpha=opacity, color='k', label='Decision Tree')
#
# ax.set_xlabel('Data Set')
# ax.set_ylabel('Macro-F1 Scores')
# #plt.title(f'')
# plt.xticks(index + bar_width, ('Original Data', 'All PCs', 'Two PCs'))
# plt.legend(loc="lower right")
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
#
# plt.tight_layout()
# plt.show()
