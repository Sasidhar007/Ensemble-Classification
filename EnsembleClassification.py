
# coding: utf-8

# In[9]:


import scipy
import csv
import pandas as p
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools


# In[2]:


sns.set(style="white")


# In[3]:


data = p.read_csv('C:\\Users\\msasi\\Desktop\\CS1\\pima-indians-diabetes.csv', delimiter=',')


# In[4]:


data.head()


# In[5]:


data['Class'].value_counts()   #Gives us the Class Distibution of the DataSet


# In[26]:


#Basic Statistics for Various Columns
notp1 = data[data['Class'] == 1]['NumberOfTimesPregnant']
notp1.describe()


# In[27]:


bi_nopt1 = data[data['Class'] == 1]
bi_nopt1['NumberOfTimesPregnant'].hist()


# In[24]:


notp0 = data[data['Class'] == 0]['NumberOfTimesPregnant']
notp0.describe()


# In[28]:


bi_nopt0 = data[data['Class'] == 0]
bi_nopt0['NumberOfTimesPregnant'].hist()


# In[53]:


#Scatterplot
plt.scatter(data['Class'],data['NumberOfTimesPregnant'])
#just made a change


# In[21]:


#Description of the attributes:
data['NumberOfTimesPregnant'].describe()


# In[22]:


data['PlasmaGlucose'].describe()


# In[15]:


data['DiastolicBloodPressure'].describe()


# In[16]:


data['DiabetesPedigreeFunction'].describe()


# In[17]:


data['2HrSerumInsulin'].describe()


# In[18]:


data['TricepsSkinFoldThickness'].describe()


# In[19]:


data['BMI'].describe()


# In[20]:


data['Age'].describe()


# In[50]:


#corr = data.corr()
data.corr()


# In[47]:


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[63]:


#p.scatter_matrix(data, alpha=0.2, diagonal='kde')


# In[62]:


sns.pairplot(data)


# In[6]:


X = data.values[:, 0:7]
Y = data.values[:,8]
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)


# In[23]:


# Decision Tree Algorithm
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100)
clf_gini.fit(x_train, y_train)


# In[8]:


from sklearn import metrics
print(clf_gini.feature_importances_)


# In[11]:


scipy.stats.chisquare(X)


# In[13]:


trainingtest = clf_gini.predict(x_test)


# In[14]:


accuracy_score(y_test, trainingtest)


# In[15]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100)
clf_entropy.fit(x_train, y_train)


# In[16]:


trainingtestentropy = clf_entropy.predict(x_test)


# In[17]:


accuracy_score(y_test, trainingtestentropy)


# In[18]:


#Decision tree with k fold cross validation
cv_results = cross_val_score(clf_gini, x_train, y_train, cv=10)


# In[19]:


print(cv_results)


# In[20]:


np.mean(cv_results)


# In[33]:


#Naive Bayes Algorithm
gnb = GaussianNB()
x_testing = gnb.fit(x_train, y_train).predict(x_test)


# In[34]:


print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != x_testing).sum()))


# In[35]:


accuracy_score(y_test,x_testing)


# In[27]:


#knearest neighbors with 10 fold cross validation
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid,cv=10)
knn_cv.fit(x_train,y_train)


# In[25]:


knn_cv.best_params_


# In[26]:


knn_cv.best_score_


# In[27]:


yhat = knn_cv.predict(x_test)


# In[28]:


yhat.std()


# In[29]:


accuracy_score(y_test,yhat)


# In[28]:


# Ensemble method: Bagging(Bootstrap Aggregating) bootstrap set to default as True
from sklearn.ensemble import BaggingClassifier
bagging1 = BaggingClassifier(knn_cv, max_samples=0.5, max_features=1)


# In[29]:


bagging1.fit(x_train,y_train)


# In[55]:


yhatensemble = bagging1.predict(x_test)


# In[56]:


#Accuracy on the test set
accuracy_score(y_test,yhatensemble)


# In[30]:


yhaten = bagging1.predict(X)


# In[31]:


#Accuracy on the whole set
accuracy_score(Y,yhaten)


# In[7]:


#ensemble method: Stacking
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier


# In[10]:


clf1 = KNeighborsClassifier(n_neighbors=30) # As we got 30 to be the best number of neighbors in the above statements
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],meta_classifier=lr)

print('10-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf],['KNN','Random Forest','Naive Bayes','StackingClassifier']):

    scores = model_selection.cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# In[24]:


#ensemble method: Boosting (AdaBoost)
from sklearn.ensemble import AdaBoostClassifier
clfAda = AdaBoostClassifier(n_estimators=100)
scoresAda = cross_val_score(clfAda, x_train, y_train)
scoresAda.mean()


# In[54]:


#predicting on test set and then comparing it with original label
clfAda.fit(x_train,y_train)
yhatada = clfAda.predict(x_test)
accuracy_score(y_test, yhatada)


# In[51]:


#on the complete data set accuracy
scoresAda1 = cross_val_score(clfAda,X,Y)
scoresAda1.mean()

