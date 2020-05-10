#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Decision Tree. You will use this classification algorithm to build a model from historical data of patients, and their response to different medications. Then you use the trained decision tree to predict the class of a unknown patient, 
#or to find a proper drug for a new patient.
#you can use the training part of the dataset to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.


# In[38]:


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[39]:


#OR USE THE URL of the CSV file given in GitHub
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"
my_data = pd.read_csv(url, delimiter=",")
my_data[0:5]


# In[40]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[41]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[42]:


y = my_data["Drug"]
y[0:5]


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


#Now train_test_split will return 4 different parameters. We will name them:
#X_trainset, X_testset, y_trainset, y_testset

#The train_test_split will need the parameters:
#X, y, test_size=0.3, and random_state=3.

#The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.


# In[45]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# In[46]:


#We will first create an instance of the DecisionTreeClassifier called drugTree.Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
#Entropy is the amount of information disorder or the amount of randomness in the data. The entropy in the node depends on how much random data is in that node and is calculated for each node. In decision trees, we're looking for trees that have the smallest entropy in their nodes. The entropy is used to calculate the homogeneity of the samples in that node. If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one.


# In[47]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[48]:


#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset


# In[49]:


drugTree.fit(X_trainset,y_trainset)


# In[50]:


#Let's make some predictions on the testing dataset and store it into a variable called predTree.


# In[51]:


predTree = drugTree.predict(X_testset)


# In[52]:


print (predTree [0:5])
print (y_testset [0:5])


# In[53]:


#Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
#In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.


# In[54]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[ ]:





# In[ ]:




