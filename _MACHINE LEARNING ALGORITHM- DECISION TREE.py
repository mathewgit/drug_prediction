#!/usr/bin/env python
# coding: utf-8

# In[252]:


#Decision Tree. 
#You will use this classification algorithm to build a model from historical data of patients, and their response to different medications. 
#Then you use the trained decision tree to predict the class of a unknown patient, #or to find a proper drug for a new patient.
#you can use the training part of the dataset to build a decision tree, and then use it to predict the 
#class of a unknown patient, or to prescribe it to a new patient.
# You will have to grow your own decision tree.
# Imagine that you're a medical researcher compiling data for a study. You've already collected data about a set of patients all of whom suffered from the same illness. During their course of treatment, each patient responded to one of two medications. We call them drug A and drug B. Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness.
# During their course of treatment, each patient responded to one of two medications. We call them drug A and drug B.
#Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness 
#Feature sets of this dataset are age, gender, blood pressure, and cholesterol of our group of patients 
# Target is the drug that each patient responded to
# Calculate the significance of an attribute to see if it's an effective attribute or not.


# In[253]:


#HOW TO BUILD A DECISION TREE LEARNING ALGORITHM.
# 1. Choose an attribute from your dataset
# 2. Calculate the significiance of attribute in splitting of data
# 3. Split data based on the value of the best attribute
# 4. Go to step 1 and repeat.

# Decision trees are built using recursive partitining to classify the data.Decision tree Algorthim chooses the most predictive feature to split the data.Determine which attribute is the best or more predictive  to split data based on the feature.
# Predictiveness is based on decrease in impurity of nodes. The Algorithim is looking for the best feature to decrease the impurity after splitting them up based on features
# Impurity of nodes is calculated by entropy of data in the node. So, what is entropy? Entropy is the amount of information disorder or the amount of randomness in the data. The entropy in the node depends on how much random data is in that node and is calculated for each node. In decision trees, we're looking for trees that have the smallest entropy in their nodes. The entropy is used to calculate the homogeneity of the samples in that node. If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one.
# We should go through all the attributes and calculate the entropy after the split and then choose the best attribute. 
#  Information gain is the information that can increase the level of certainty after splitting. It is the entropy of a tree before the split minus the weighted entropy after the split by an attribute. We can think of information gain and entropy as opposites. As entropy or the amount of randomness decreases, the information gain or amount of certainty increases and vice versa. So, constructing a decision tree is all about finding attributes that return the highest information gain. 


# In[254]:


#IMPORT THE REQUIRED LIBRARIES
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[255]:


#Pre-processing
#Using my_data as the Drug.csv data read by pandas, declare the following variables:

#X as the Feature Matrix (data of my_data)
#y as the response vector (target)
#Remove the column containing the target name since it doesn't contain numeric values.


# In[256]:


#OR USE THE URL of the CSV file given in GitHub.
# The sample is a bnary classifier and we can predict the class of an unknown patient.(Which drug to prescribe to a new patient)
url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"
my_data = pd.read_csv(url, delimiter=",")
my_data[0:5]
my_data


# In[257]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# In[258]:


# As you may figure out, some features in this dataset are categorical such as Sex or BP. 
#Unfortunately, Sklearn Decision Trees do not handle categorical variables. 
#But still we can convert these features to numerical values. pandas.get_dummies() Convert categorical variable into dummy/indicator variables.

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


# In[259]:


#Setting up the Decision Tree
#We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.

y = my_data["Drug"]
y[0:5]


# In[187]:


from sklearn.model_selection import train_test_split


# In[188]:


#Now train_test_split will return 4 different parameters. We will name them:
#X_trainset, X_testset, y_trainset, y_testset

#The train_test_split will need the parameters:
#X, y, test_size=0.3, and random_state=3.

#The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits.


# In[190]:


#test_sizefloat or int, default=None
#If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.

#train_sizefloat or int, default=None
#If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size

#random_stateint or RandomState instance, default=None
#Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.


# In[205]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.40, random_state=5)


# In[206]:


X_trainset


# In[207]:


y_testset.count()


# In[250]:


# MODELING

#We will first create an instance of the DecisionTreeClassifier called drugTree.
#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
#Entropy is the amount of information disorder or the amount of randomness in the data. 
#The entropy in the node depends on how much random data is in that node and is calculated for each node. 
#In decision trees, we're looking for trees that have the smallest entropy in their nodes.
#The entropy is used to calculate the homogeneity of the samples in that node. 
#If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one.

# the method uses recursive partitioning to split the training records into segments by minimizing the impurity at each step. 
#Impurity of nodes is calculated by entropy of data in the node. 
#So, what is entropy? Entropy is the amount of information disorder or the amount of randomness in the data. 
#The entropy in the node depends on how much random data is in that node and is calculated for each node. 
#In decision trees, we're looking for trees that have the smallest entropy in their nodes. The entropy is used to calculate the homogeneity of the samples in that node. If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one.
#Entropy is the amount of information disorder or the amount of randomness in the data. 
#The entropy in the node depends on how much random data is in that node and is calculated for each node. 
#In decision trees, we're looking for trees that have the smallest entropy in their nodes. 
#The entropy is used to calculate the homogeneity of the samples in that node. 
#If the samples are completely homogeneous, the entropy is zero and if the samples are equally divided it has an entropy of one.

# max_depth:int, default=None
#The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

#criterion{“gini”, “entropy”}, default=”gini”
#The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.


# In[251]:


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 10)
drugTree # it shows the default parameters


# In[235]:


#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset


# In[236]:


drugTree.fit(X_trainset,y_trainset)


# In[237]:


#Let's make some predictions on the testing dataset and store it into a variable called predTree.


# In[245]:


predTree = drugTree.predict(X_testset)
predTree 


# In[247]:


y_testset


# In[248]:


#Accuracy classification score computes subset accuracy: the set of labels predicted 
#for a sample must exactly match the corresponding set of labels in y_true.
#In multilabel classification, the function returns the subset accuracy. 
#If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.


# In[249]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[ ]:


get_ipython().system('conda install -c conda-forge pydotplus -y')
get_ipython().system('conda install -c conda-forge python-graphviz -y')


# In[232]:


# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[233]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:




