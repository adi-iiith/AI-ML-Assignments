#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import seaborn as sns
import pprint
from collections import OrderedDict


# ##   Part 1 :  Categorical Data <br>

# In[24]:


# loading the data in a dataframe
# train_set and validate_set  80:20

file = 'train.csv'
df = pd.read_csv(file)
x = np.unique(df['left'], return_counts = True)

c_list = ['Work_accident','promotion_last_5years','sales','salary','left']

df1 = df[c_list]
train_set = df1.sample(frac=0.8,random_state=200)
validate_set = df1.drop(train_set.index)
x = np.unique(train_set['left'], return_counts = True)
# print(x)
c_list.remove('left')

# -------------------------------------------------------------------------------

def entropy_(df):
    total = df['left'].count()
    positive = df[df['left'] == 1]
    pos_c = positive['left'].count()
    neg_c = total - pos_c
    if pos_c == 0 or neg_c == 0 :
        return 0
    ent = - (pos_c/total * math.log(pos_c/total,2) 
                 + neg_c/total * math.log(neg_c/total,2))
    return ent
# --------------------------------------------------------------------------------- IG

def IG(data,attribute,entropy):
    isum = 0
    t = len(data.index)
    label = (data[attribute].unique()).tolist()
    for i in label:
        sdf = data[data[attribute] == i]
        isum = isum + ((len(sdf.index)/t) * entropy_(sdf))
    return entropy - isum

# --------------------------------------------------------------------------------- Best Attribute

def best_attribute(df,attribute):
    e = entropy_(df)
    attribute_list = {}
    for best in attribute :
        attribute_list[best] = IG(df , best , e)
    b = max(attribute_list, key=attribute_list.get)
    return b
# --------------------------------------------------------------------------------- tree

def tree(df , attribute_list , pd , root=None):
    if len(np.unique(df['left'])) <=1 :
        return {'predict': np.unique(df['left'])[0]}
    elif len(attribute_list) == 0:
        t = np.argmax(np.unique(df['left'], return_counts = True)[1])
        return {'predict': np.unique(df['left'])[t]}
    elif len(df) == 0:
        t = np.argmax(np.unique(pd['left'], return_counts = True)[1])
        return {'predict': np.unique(pd['left'])[t]}
    b = best_attribute(df,attribute_list[:])
    attribute_list.remove(b)
    # tree empty
    if root is None:
        root = {}
        root[b] = {}
    for i in df[b].unique():
        new_df = df[df[b]==i]
        root[b][i] = tree( new_df , attribute_list[:] , df)
    return root

# ---------------------------------------------------------------------------------

# Build Tree ------ Call the function 

root = tree(train_set,c_list[:],train_set)
# print(root)
prediction = []   # List of prediction to store a list of [ 0 ,1 ] of the predicted value on Validate Set

# --------------------------------------------------------------------------------- Compare 

def compare(row,root):
    try:
        if list(root.keys())[0] == 'predict':
            return root['predict']
        else:
            value = row[list(root.keys())[0]]
            return compare(row,root[list(root.keys())[0]][value])
    except:
        return 0            # Maximum values in DataSet is 0 so return 0 

# ---------------------------------------------------------------------------------  Test 

def test(root,df):
    for index,row in df.iterrows():
        prediction.append(compare(row,root))

        
        
# ---------------------------------------------------------------------------------  Validate Set

def final_prediction_list():
    prediction_list= OrderedDict()
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(actual)):
        if prediction[i] == actual[i]:
            if prediction[i] == 1:
                tp += 1
            elif prediction[i] == 0:
                tn += 1
        else:
            if actual[i] == 1:
                fn += 1
            elif actual[i] == 0:
                fp += 1
    prediction_list['True Positive'] = tp
    prediction_list['False Positive'] = fp
    prediction_list['True Negative'] = tn
    prediction_list['False Negative'] = fn
    accuracy = (tp+tn)/len(actual)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2/(1/recall + 1/precision)
    prediction_list['Accuracy'] = accuracy
    prediction_list['Precision'] = precision
    prediction_list['Recall'] = recall
    prediction_list['F1_Score'] = f1_score
    pp = pprint.PrettyPrinter(indent=6)
    for i, (key, value) in enumerate(prediction_list.items()):
        print(key + ":: ", value)
        print("")
# ---------------------------------------------------------------------------------

actual = validate_set['left'].tolist()

# test the data , call test function

test(root,validate_set)
final_prediction_list()


# # Part 2 : Numerica Data included in Train_Set <br>
# 

# In[25]:


# loading the data in a dataframe
file = 'train.csv'
df = pd.read_csv(file)
x = np.unique(df['left'], return_counts = True)
c_list = ['satisfaction_level','last_evaluation','number_project','average_montly_hours',
          'time_spend_company','Work_accident','promotion_last_5years','sales','salary','left']
df1 = df[c_list]

train_set = df1.sample(frac=0.8,random_state=200)
validate_set = df1.drop(train_set.index)
x = np.unique(train_set['left'], return_counts = True)
c_list.remove('left')

# ---------------------------------------------------------------------------------

def entropy_(df):
    total = df['left'].count()
    positive = df[df['left'] == 1]
    pos_c = positive['left'].count()
    neg_c = total - pos_c
    if pos_c == 0 or neg_c == 0 :
        return 0
    ent = - (pos_c/total * math.log(pos_c/total,2) 
                 + neg_c/total * math.log(neg_c/total,2))
    return ent
    
# ---------------------------------------------------------------------------------

def IG(data,attribute,entropy):
    isum = 0
    t = len(data.index)
    label = (data[attribute].unique()).tolist()
    for i in label:
        sdf = data[data[attribute] == i]
        isum = isum + ((len(sdf.index)/t) * entropy_(sdf))
    return entropy - isum
# ---------------------------------------------------------------------------------
# best attribute

def best_attribute(df,attribute):
    e = entropy_(df)
    attribute_list = {}
    for best in attribute :
        attribute_list[best] = IG(df , best , e)
    b = max(attribute_list, key=attribute_list.get)
    return b
# ---------------------------------------------------------------------------------

numeric_data = ['satisfaction_level','last_evaluation','number_project',
                'average_montly_hours','time_spend_company']
attribute_dict = {}
# --------------------------------------------------------------------------------- 
# Preprocess the numeric data accordingly

def convert(df,attr):
    mapp = {}
    check = []
    avg = []
    label = 0
    llist = []
    sortd = df.sort_values(attr)
    uv = sortd[attr].unique().tolist()
    for i in uv:
        temp = sortd[sortd[attr] == i]
        mapp[i] = np.unique(temp['left'])[np.argmax(np.unique(temp['left'],return_counts=True)[1])]
        
    prev = uv[0]
    h = mapp[uv[0]]
    for i in uv[1:len(uv)]:
        if mapp[i] == h:
            prev =i
            continue
        else :
            h = 1-h
            check.append(prev)
            prev = i
            check.append(i)
    for i in range(0,len(check),2):
        avg.append((check[i]+check[i+1])/2)
    attribute_dict[attr] = avg
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df
# --------------------------------------------------------------------------------- 

def tree(df , attribute_list , pd , root=None):
    if len(np.unique(df['left'])) <=1 :
        return {'predict': np.unique(df['left'])[0]}
    elif len(attribute_list) == 0:
        t = np.argmax(np.unique(df['left'], return_counts = True)[1])
        return {'predict': np.unique(df['left'])[t]}
    elif len(df) == 0:
        t = np.argmax(np.unique(pd['left'], return_counts = True)[1])
        return {'predict': np.unique(pd['left'])[t]}
    b = best_attribute(df,attribute_list[:])
    attribute_list.remove(b)
    # tree empty
    if root is None:
        root = {}
        root[b] = {}
    for i in df[b].unique():
        new_df = df[df[b]==i]
        root[b][i] = tree( new_df , attribute_list[:] , df)
    return root
# print (c_list)
# --------------------------------------------------------------------------------- 
# BUILD TREE
# --------------------------------------------------------------------------------- 

# Assign the train set on the new data

for i in numeric_data:
    train_set = convert(train_set,i)

# --------------------------------------------------------------------------------- 

root = tree(train_set,c_list[:],train_set)
prediction = []

# print(root)

# --------------------------------------------------------------------------------- 

def prediction_convert(df,attr):
    avg = attribute_dict[attr]
    label = 0
    llist =[]
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df


# --------------------------------------------------------------------------------- 

def compare(row,root):
    try:
        if list(root.keys())[0] == 'predict':
            return root['predict']
        else:
            value = row[list(root.keys())[0]]
            return compare(row,root[list(root.keys())[0]][value])
    except:
        return 0         
    
# --------------------------------------------------------------------------------- 

  
def test(root,df):
    for index,row in df.iterrows():
        prediction.append(compare(row,root))

# --------------------------------------------------------------------------------- 

def final_prediction_list():
    prediction_list= OrderedDict()
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(actual)):
        if prediction[i] == actual[i]:
            if prediction[i] == 1:
                tp += 1
            elif prediction[i] == 0:
                tn += 1
        else:
            if actual[i] == 1:
                fn += 1
            elif actual[i] == 0:
                fp += 1
    prediction_list['True Positive'] = tp
    prediction_list['False Positive'] = fp
    prediction_list['True Negative'] = tn
    prediction_list['False Negative'] = fn
    accuracy = (tp+tn)/len(actual)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2/(1/recall + 1/precision)
    prediction_list['Accuracy'] = accuracy
    prediction_list['Precision'] = precision
    prediction_list['Recall'] = recall
    prediction_list['F1_Score'] = f1_score
    pp = pprint.PrettyPrinter(indent=6)
    for i, (key, value) in enumerate(prediction_list.items()):
        print(key + ":: ", value)
        print("")
        
# --------------------------------------------------------------------------------- 


# --------------------------------------------------------------------------------- 
# Assign the validate set on the new data

for i in numeric_data:
    validate_set = prediction_convert(validate_set,i)
# ---------------------------------------------------------------------------------

# for i in numeric_data:
#     test_set = convert(test_set,i)

# ---------------------------------------------------------------------------------

test(root,validate_set)
actual = validate_set['left'].tolist()
final_prediction_list()


# # Part 3.2 : Using Gini Index <br>
#  
# <h6> Part 3.1 : { Using Entropy } Same as Part 2</h6> <br>

# In[26]:


# loading the data in a dataframe
file = 'train.csv'
df = pd.read_csv(file)
x = np.unique(df['left'], return_counts = True)
c_list = ['satisfaction_level','last_evaluation','number_project','average_montly_hours',
          'time_spend_company','Work_accident','promotion_last_5years','sales','salary','left']
df1 = df[c_list]

train_set = df1.sample(frac=0.8,random_state=200)
validate_set = df1.drop(train_set.index)
x = np.unique(train_set['left'], return_counts = True)
c_list.remove('left')

# ---------------------------------------------------------------------------------

def gini_index(df):
    total = df['left'].count()
    positive = df[df['left'] == 1]
    pos_c = positive['left'].count()
    neg_c = total - pos_c
# In [26]:

    if pos_c == 0 or neg_c == 0 :
        return 0
    ent = - (math.pow(pos_c/total,2) 
                 + math.pow(neg_c/total,2))
    return ent
    
# ---------------------------------------------------------------------------------

def IG(data,attribute,entropy):
    isum = 0
    t = len(data.index)
    label = (data[attribute].unique()).tolist()
    for i in label:
        sdf = data[data[attribute] == i]
        isum = isum + ((len(sdf.index)/t) * gini_index(sdf))
    return entropy - isum
# ---------------------------------------------------------------------------------
# best attribute

def best_attribute(df,attribute):
    e = gini_index(df)
    attribute_list = {}
    for best in attribute :
        attribute_list[best] = IG(df , best , e)
    b = max(attribute_list, key=attribute_list.get)
    return b
# ---------------------------------------------------------------------------------

numeric_data = ['satisfaction_level','last_evaluation','number_project',
                'average_montly_hours','time_spend_company']
attribute_dict = {}
# --------------------------------------------------------------------------------- 
# Preprocess the numeric data accordingly

def convert(df,attr):
    mapp = {}
    check = []
    avg = []
    label = 0
    llist = []
    sortd = df.sort_values(attr)
    uv = sortd[attr].unique().tolist()
    for i in uv:
        temp = sortd[sortd[attr] == i]
        mapp[i] = np.unique(temp['left'])[np.argmax(np.unique(temp['left'],return_counts=True)[1])]
        
    prev = uv[0]
    h = mapp[uv[0]]
    for i in uv[1:len(uv)]:
        if mapp[i] == h:
            prev =i
            continue
        else :
            h = 1-h
            check.append(prev)
            prev = i
            check.append(i)
    for i in range(0,len(check),2):
        avg.append((check[i]+check[i+1])/2)
    attribute_dict[attr] = avg
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df
# --------------------------------------------------------------------------------- 

def tree(df , attribute_list , pd , root=None):
    if len(np.unique(df['left'])) <=1 :
        return {'predict': np.unique(df['left'])[0]}
    elif len(attribute_list) == 0:
        t = np.argmax(np.unique(df['left'], return_counts = True)[1])
        return {'predict': np.unique(df['left'])[t]}
    elif len(df) == 0:
        t = np.argmax(np.unique(pd['left'], return_counts = True)[1])
        return {'predict': np.unique(pd['left'])[t]}
    b = best_attribute(df,attribute_list[:])
    attribute_list.remove(b)
    # tree empty
    if root is None:
        root = {}
        root[b] = {}
    for i in df[b].unique():
        new_df = df[df[b]==i]
        root[b][i] = tree( new_df , attribute_list[:] , df)
    return root
# print (c_list)
# --------------------------------------------------------------------------------- 
# BUILD TREE
# --------------------------------------------------------------------------------- 

# Assign the train set on the new data

for i in numeric_data:
    train_set = convert(train_set,i)

# --------------------------------------------------------------------------------- 

root = tree(train_set,c_list[:],train_set)
prediction = []

# print(root)

# --------------------------------------------------------------------------------- 

def prediction_convert(df,attr):
    avg = attribute_dict[attr]
    label = 0
    llist =[]
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df


# --------------------------------------------------------------------------------- 

def compare(row,root):
    try:
        if list(root.keys())[0] == 'predict':
            return root['predict']
        else:
            value = row[list(root.keys())[0]]
            return compare(row,root[list(root.keys())[0]][value])
    except:
        return 0         
    
# --------------------------------------------------------------------------------- 

  
def test(root,df):
    for index,row in df.iterrows():
        prediction.append(compare(row,root))

# --------------------------------------------------------------------------------- 

def final_prediction_list():
    prediction_list= OrderedDict()
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(actual)):
        if prediction[i] == actual[i]:
            if prediction[i] == 1:
                tp += 1
            elif prediction[i] == 0:
                tn += 1
        else:
            if actual[i] == 1:
                fn += 1
            elif actual[i] == 0:
                fp += 1
    prediction_list['True Positive'] = tp
    prediction_list['False Positive'] = fp
    prediction_list['True Negative'] = tn
    prediction_list['False Negative'] = fn
    accuracy = (tp+tn)/len(actual)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2/(1/recall + 1/precision)
    prediction_list['Accuracy'] = accuracy
    prediction_list['Precision'] = precision
    prediction_list['Recall'] = recall
    prediction_list['F1_Score'] = f1_score
    pp = pprint.PrettyPrinter(indent=6)
    for i, (key, value) in enumerate(prediction_list.items()):
        print(key + ":: ", value)
        print("")
        
# --------------------------------------------------------------------------------- 


# --------------------------------------------------------------------------------- 
# Assign the validate set on the new data

for i in numeric_data:
    validate_set = prediction_convert(validate_set,i)
# ---------------------------------------------------------------------------------

# for i in numeric_data:
#     test_set = convert(test_set,i)

# ---------------------------------------------------------------------------------

test(root,validate_set)
actual = validate_set['left'].tolist()
final_prediction_list()


# # Part 3.3 : Using Missclassification Rate<br>

# In[27]:


# loading the data in a dataframe
file = 'train.csv'
df = pd.read_csv(file)
x = np.unique(df['left'], return_counts = True)
c_list = ['satisfaction_level','last_evaluation','number_project','average_montly_hours',
          'time_spend_company','Work_accident','promotion_last_5years','sales','salary','left']
df1 = df[c_list]

train_set = df1.sample(frac=0.8,random_state=200)
validate_set = df1.drop(train_set.index)
x = np.unique(train_set['left'], return_counts = True)
c_list.remove('left')

# ---------------------------------------------------------------------------------

def miss_rate(df):
    total = df['left'].count()
    positive = df[df['left'] == 1]
    pos_c = positive['left'].count()
    neg_c = total - pos_c
    if pos_c == 0 or neg_c == 0 :
        return 0
    ent =  min((pos_c/total) 
                 ,(neg_c/total))
    return ent
    
    
# ---------------------------------------------------------------------------------

def IG(data,attribute,entropy):
    isum = 0
    t = len(data.index)
    label = (data[attribute].unique()).tolist()
    for i in label:
        sdf = data[data[attribute] == i]
        isum = isum + ((len(sdf.index)/t) * miss_rate(sdf))
    return entropy - isum
# ---------------------------------------------------------------------------------
# best attribute

def best_attribute(df,attribute):
    e = miss_rate(df)
    attribute_list = {}
    for best in attribute :
        attribute_list[best] = IG(df , best , e)
    b = max(attribute_list, key=attribute_list.get)
    return b
# ---------------------------------------------------------------------------------

numeric_data = ['satisfaction_level','last_evaluation','number_project',
                'average_montly_hours','time_spend_company']
attribute_dict = {}
# --------------------------------------------------------------------------------- 
# Preprocess the numeric data accordingly

def convert(df,attr):
    mapp = {}
    check = []
    avg = []
    label = 0
    llist = []
    sortd = df.sort_values(attr)
    uv = sortd[attr].unique().tolist()
    for i in uv:
        temp = sortd[sortd[attr] == i]
        mapp[i] = np.unique(temp['left'])[np.argmax(np.unique(temp['left'],return_counts=True)[1])]
        
    prev = uv[0]
    h = mapp[uv[0]]
    for i in uv[1:len(uv)]:
        if mapp[i] == h:
            prev =i
            continue
        else :
            h = 1-h
            check.append(prev)
            prev = i
            check.append(i)
    for i in range(0,len(check),2):
        avg.append((check[i]+check[i+1])/2)
    attribute_dict[attr] = avg
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df
# --------------------------------------------------------------------------------- 

def tree(df , attribute_list , pd , root=None):
    if len(np.unique(df['left'])) <=1 :
        return {'predict': np.unique(df['left'])[0]}
    elif len(attribute_list) == 0:
        t = np.argmax(np.unique(df['left'], return_counts = True)[1])
        return {'predict': np.unique(df['left'])[t]}
    elif len(df) == 0:
        t = np.argmax(np.unique(pd['left'], return_counts = True)[1])
        return {'predict': np.unique(pd['left'])[t]}
    b = best_attribute(df,attribute_list[:])
    attribute_list.remove(b)
    # tree empty
    if root is None:
        root = {}
        root[b] = {}
    for i in df[b].unique():
        new_df = df[df[b]==i]
        root[b][i] = tree( new_df , attribute_list[:] , df)
    return root
# print (c_list)
# --------------------------------------------------------------------------------- 
# BUILD TREE
# --------------------------------------------------------------------------------- 

# Assign the train set on the new data

for i in numeric_data:
    train_set = convert(train_set,i)

# --------------------------------------------------------------------------------- 

root = tree(train_set,c_list[:],train_set)
prediction = []

# print(root)

# --------------------------------------------------------------------------------- 

def prediction_convert(df,attr):
    avg = attribute_dict[attr]
    label = 0
    llist =[]
    for i in df.index:
        attv = df.loc[i,attr]
        label = 0
        f = 0
        for a in avg:
            if attv < a:
                f = 1
                llist.append(label)
                break
            else :
                label += 1
        if f == 0:
            llist.append(label)
    
    df = df.drop(attr,axis=1)           ##
    new1 = pd.Series(llist)              ##
    df[attr] = new1.values
    return df


# --------------------------------------------------------------------------------- 

def compare(row,root):
    try:
        if list(root.keys())[0] == 'predict':
            return root['predict']
        else:
            value = row[list(root.keys())[0]]
            return compare(row,root[list(root.keys())[0]][value])
    except:
        return 0         
    
# --------------------------------------------------------------------------------- 

  
def test(root,df):
    for index,row in df.iterrows():
        prediction.append(compare(row,root))

# --------------------------------------------------------------------------------- 

def final_prediction_list():
    prediction_list= OrderedDict()
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(actual)):
        if prediction[i] == actual[i]:
            if prediction[i] == 1:
                tp += 1
            elif prediction[i] == 0:
                tn += 1
        else:
            if actual[i] == 1:
                fn += 1
            elif actual[i] == 0:
                fp += 1
    prediction_list['True Positive'] = tp
    prediction_list['False Positive'] = fp
    prediction_list['True Negative'] = tn
    prediction_list['False Negative'] = fn
    accuracy = (tp+tn)/len(actual)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2/(1/recall + 1/precision)
    prediction_list['Accuracy'] = accuracy
    prediction_list['Precision'] = precision
    prediction_list['Recall'] = recall
    prediction_list['F1_Score'] = f1_score
    pp = pprint.PrettyPrinter(indent=6)
    for i, (key, value) in enumerate(prediction_list.items()):
        print(key + ":: ", value)
        print("")
        
# --------------------------------------------------------------------------------- 


# --------------------------------------------------------------------------------- 
# Assign the validate set on the new data

for i in numeric_data:
    validate_set = prediction_convert(validate_set,i)
# ---------------------------------------------------------------------------------

# for i in numeric_data:
#     test_set = convert(test_set,i)

# ---------------------------------------------------------------------------------

test(root,validate_set)
actual = validate_set['left'].tolist()
final_prediction_list()


# # Part 4 : Plot Data ( based on max IG value) <br>

# In[34]:


# loading the data in a dataframe
file = 'train.csv'
df = pd.read_csv(file)

def entropy_(df):
    total = df['left'].count()
    positive = df[df['left'] == 1]
    pos_c = positive['left'].count()
    neg_c = total - pos_c
    if pos_c == 0 or neg_c == 0 :
        return 0
    ent = - (pos_c/total * math.log(pos_c/total,2) 
                 + neg_c/total * math.log(neg_c/total,2))
    return ent
    
# entropy_(df1)

# ------------------------------------------------------------------------

def IG(data,attribute,entropy):
    isum = 0
    t = len(data.index)
    label = (data[attribute].unique()).tolist()
    for i in label:
        sdf = data[data[attribute] == i]
        isum = isum + ((len(sdf.index)/t) * entropy_(sdf))
    return entropy - isum

info_gain = {}
e = entropy_(df)

for i in c_list:
    info_gain[i]=(IG(df,i,e))

pprint.pprint(sorted(info_gain.items(), key=operator.itemgetter(1)))

# max values -- satisfaction_level and number_projects 
# drop other attributes from DF

c_list = ['last_evaluation','average_montly_hours','time_spend_company',
          'Work_accident','promotion_last_5years','sales','salary']
df = df.drop(c_list,axis=1)


# In[37]:


left=df['left'].values
satisfaction = df['satisfaction_level'].values
project = df['number_project'].values

sns.set_style("whitegrid");
sns.set(font_scale=2)
ax = sns.FacetGrid(df, hue="left", height=15)    .map(plt.scatter, "satisfaction_level", "number_project")    .add_legend();
ax.set(xlabel='Satisfaction Level', ylabel='Number of Projects')

plt.show();


# # Part 5 : Plot Data ( Training and Validation Error wrt Depth and Nodes ) <br>

# In[ ]:





# # Part 6 : Explain how decision tree is suitable handle missing values<br>

# In[ ]:




