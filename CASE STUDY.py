#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study (Upgrad Assignment Machiene Learning)
# 

# Problem Statement
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# 
#  
# 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%. 
# 
#  
# 
# Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as ‘Hot Leads’. If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel:
# 
# Lead Conversion Process - Demonstrated as a funnel
# Lead Conversion Process - Demonstrated as a funnel
# As you can see, there are a lot of leads generated in the initial stage (top) but only a few of them come out as paying customers from the bottom. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion.
# 
#  
# 
# X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# #### resolving steps:
# 
# * **Steps will be as follows:**
#     * Importing the data
#     * Data Cleaning
#     * EDA
#     * Model Building
#     * Model Evaluation
#     * Summary
# 

# In[126]:


# Importing Libraries 

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[127]:


#set the screen 

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.set_option('display.width',200)
pd.set_option('display.html.border',1)

pd.set_option('display.max_columns',200)
pd.set_option('display.max_colwidth',255)
pd.set_option('display.max_info_columns',200)
pd.set_option('display.max_info_rows',200)


# ## Importing the data

# In[128]:


x_edu_df = pd.read_csv("Leads.csv")
x_edu_df.head(4)


# In[129]:


#lets check the shape of the data
x_edu_df.shape


# In[130]:


#describe the data
x_edu_df.describe()


# In[131]:


#type of the data
x_edu_df.info()


# In[132]:


#convert in to float TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'
x_edu_df['TotalVisits'] = x_edu_df['TotalVisits'].astype(float)
x_edu_df['Total Time Spent on Website'] = x_edu_df['Total Time Spent on Website'].astype(float)
x_edu_df['Page Views Per Visit'] = x_edu_df['Page Views Per Visit'].astype(float)


# In[133]:


#fill the blank value with mean 
x_edu_df['TotalVisits'].fillna(x_edu_df['TotalVisits'].mean(), inplace=True)
x_edu_df['Total Time Spent on Website'].fillna(x_edu_df['Total Time Spent on Website'].mean(), inplace=True)
x_edu_df['Page Views Per Visit'].fillna(x_edu_df['Page Views Per Visit'].mean(), inplace=True)

#check the null value
x_edu_df.isnull().sum()


# In[134]:


#lets check null values percentage
round(100*(x_edu_df.isnull().sum()/len(x_edu_df.index)), 2)


# In[135]:


#data cleaning replace Select  values with null values
x_edu_df = x_edu_df.replace('Select', np.nan)


# In[136]:


#data columns
x_edu_df.columns


# #### Lets divide data into categorical and numerical and target columns and perform EDA

# In[137]:


#select all categorical columns
cat_cols = x_edu_df.select_dtypes(include=['object']).columns
cat_cols


# In[138]:


#describe categorical columns
x_edu_df[cat_cols].describe()


# In[139]:


#devide top freq value of cat columns with total rows
for col in cat_cols:
    print(col,":",round(x_edu_df[col].value_counts(normalize=True,dropna=False)[0]*100,2),"%")


# In[140]:


target_column = {'Converted'}
#make two catalogues for categorical columns drop_col and selected_col

#add drop_col with 95% top freq value
drop_col={}
for col in cat_cols:
    if round(x_edu_df[col].value_counts(normalize=True,dropna=False)[0]*100,2) > 95:
        drop_col[col]=round(x_edu_df[col].value_counts(normalize=True,dropna=False)[0]*100,2)
#add selected_col with other columns
selected_col={}
for col in cat_cols:
    if round(x_edu_df[col].value_counts(normalize=True,dropna=False)[0]*100,2) < 95:
        selected_col[col]=round(x_edu_df[col].value_counts(normalize=True,dropna=False)[0]*100,2)
#print drop_col and selected_col
print(f"drop_col: {drop_col.keys()}")
print("*******************")
print(f"selected_col: {selected_col.keys()}")



# In[141]:


#lets check null values percentage in categorical columns
round(100*(x_edu_df[cat_cols].isnull().sum()/len(x_edu_df.index)), 2)


# In[142]:


#lets get some inside from the selected columns
#check country wise total count
#make multiple subplots two in row
fig, ax = plt.subplots(2,2,figsize=(20,10))

#make count stacked bar plot for country and Converted count 
sns.countplot(x='Country', hue='Converted', data=x_edu_df, ax=ax[0,0])
ax[0,0].set_title('Country wise Converted count')
#rotate x axis label
for tick in ax[0,0].get_xticklabels():
    tick.set_rotation(90)
#second subplot
#make count stacked bar plot for Specialization and Converted count
sns.countplot(x='Specialization', hue='Converted', data=x_edu_df, ax=ax[0,1])
ax[0,1].set_title('Specialization wise Converted count')
#rotate x axis label
for tick in ax[0,1].get_xticklabels():
    tick.set_rotation(90)
#third subplot
#make count stacked bar plot for What is your current occupation and Converted count
sns.countplot(x='What is your current occupation', hue='Converted', data=x_edu_df, ax=ax[1,0])
ax[1,0].set_title('What is your current occupation wise Converted count')
#rotate x axis label
for tick in ax[1,0].get_xticklabels():
    tick.set_rotation(90)
#fourth subplot
#make count stacked bar plot for Lead Source and Converted count
sns.countplot(x='Lead Source', hue='Converted', data=x_edu_df, ax=ax[1,1])
ax[1,1].set_title('Lead Source wise Converted count')
#rotate x axis label
for tick in ax[1,1].get_xticklabels():
    tick.set_rotation(90)




# In[143]:


#lets finilize the list of required cat columns
cat_cols={'Last Notable Activity', 'A free copy of Mastering The Interview', 'Do Not Email', 'Lead Source', 'Last Activity', 'Lead Origin'}
drop_col={'Get updates on DM Content', 'Lead Number', 'I agree to pay the amount through cheque', 'Do Not Call', 'Receive More Updates About Our Courses', 'Magazine', 'Newspaper Article', 
'Prospect ID', 'Newspaper', 'Update me on Supply Chain Content', 'Through Recommendations', 'Search', 'Digital Advertisement', 'X Education Forums'}
#list of numerical columns
numerical_column = {'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'}


# In[144]:


#final data frame

x_edu_df=x_edu_df[list(numerical_column|cat_cols|target_column)]

print(x_edu_df.head(2))
#print null value
print(x_edu_df.isnull().sum())


# In[145]:


#lets check null values percentage in final data frame
round(100*(x_edu_df.isnull().sum()/len(x_edu_df.index)), 2)


# In[146]:


#handling null values
#cat columns
for col in cat_cols:
    x_edu_df[col].fillna(x_edu_df[col].mode()[0], inplace=True)
#numerical columns
for col in numerical_column:
    x_edu_df[col].fillna(x_edu_df[col].mean(), inplace=True)

#lets check null values percentage in final data frame
round(100*(x_edu_df.isnull().sum()/len(x_edu_df.index)), 2)



# ## Exploratory Data Analysis

# In[147]:


#make pie chart for Converted
plt.figure(figsize=(10,10))
x_edu_df['Converted'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Converted')
#dark blank font color
plt.rcParams.update({'font.size': 15, 'font.weight': 'bold', 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})

plt.show()


# In[148]:


# function for integer columns box plot and histogram and scatter plot
def plot_box_hist(df, col):
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    sns.boxplot(df[col], ax=ax[0])
    sns.distplot(df[col], ax=ax[1])
    sns.scatterplot(x=col, y='Converted', data=df, ax=ax[2])
    plt.show()


# In[149]:


# function for categorical columns count plot
def plot_count(df, col):
    plt.figure(figsize=(25, 5))
    sns.countplot(df[col])
    #white background
    plt.style.use('seaborn-whitegrid')
    plt.xticks(rotation=90)
    plt.show()
    


# In[150]:


#plot the box plot and histogram for numerical columns
for col in numerical_column:
    plot_box_hist(x_edu_df, col)

    


# In[151]:


#categorical columns
for col in cat_cols:
    plot_count(x_edu_df, col)
    


# In[152]:


def int_converted(c1,c2):
    plt.figure(figsize=(4, 4))
    ax = sns.boxplot(x=c1, y=c2, data=x_edu_df)
    
    plt.show()



for c in numerical_column:
  int_converted('Converted',c)


# In[153]:


def cat_converted(c1,c2):
  sns.catplot(col=c1 ,y=c2, data=x_edu_df, kind="count")
  plt.show()

for c in cat_cols:
  cat_converted('Converted',c)


# In[154]:


#correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(x_edu_df.corr(), annot=True, cmap='RdYlGn')
plt.show()


# In[155]:


#finilize variable for model building
x_edu_df.head()


# ## Creating Dummy variables

# In[156]:


#creating dummy variables for categorical columns
df_cat_dummy=pd.get_dummies(x_edu_df[cat_cols], drop_first=True)
df_cat_dummy.describe()


# #### plot a correlation matrix to see the correlation between the variables

# In[157]:


#correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(df_cat_dummy.corr(),  cmap='YlGnBu')
plt.show()


# In[158]:


#some of points are highly correlated to each other will drop later at the time of model building


# In[159]:


#add dummy columns to final data frame
x_edu_df=pd.concat([x_edu_df,df_cat_dummy],axis=1)
#drop orignal categorical columns
x_edu_df=x_edu_df.drop(cat_cols,axis=1)
x_edu_df.head(2)


# In[160]:


x_edu_df.shape


# ## Outlier Treatment of numerical columns

# In[161]:


#add numeric columns to df_cat_dummy
x_edu_df.shape


# In[162]:


df_cat_dummy.shape


# In[163]:


#there are some outlier we can remove those +3 std and -3 std

#check for outlier
outlier=np.abs(stats.zscore(x_edu_df))

# Outlier +3
df_outlier=x_edu_df[(outlier>3).any(axis=1)]

x_edu_df=x_edu_df[(outlier<3).all(axis=1)]
x_edu_df.shape


# ##  Normalising of continuous variables

# In[164]:


#  Normalising of continuous variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#numariacal columns
num_col=['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
#fit and transform
x_edu_df[num_col] = scaler.fit_transform(x_edu_df[num_col])
x_edu_df.head(2)


# In[165]:


x_edu_df.shape


# ## Building the model
# 

# In[166]:


#splitting the data into train and test
from sklearn.model_selection import train_test_split
# Putting feature variable to X
X = x_edu_df.drop(['Converted'], axis=1)
# Putting response variable to y
y = x_edu_df['Converted']
#splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

#lets check the shape of train and test data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# #### lets build the model using RFE becuase we have lots of variables filter first
# 

# In[167]:


# Running RFE with the output number of the variable equal 
from turtle import st


lm = LogisticRegression(max_iter=1000, random_state=100, class_weight='balanced')
lm.fit(X_train, y_train)
rfe = RFE(estimator=lm, n_features_to_select=20)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

#col with rfe support true
col = X_train.columns[rfe.support_]
#col with rfe support false
nots = X_train.columns[~rfe.support_]

#print both col
print(col)
print(nots)



# In[168]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[169]:


# RFE selected  some of indicators are not stable selected below features after running model 
final_col=['Total Time Spent on Website', 'TotalVisits', 'Last Activity_Page Visited on Website', 'Last Activity_Email Opened', 
'Lead Source_Olark Chat', 'Last Notable Activity_SMS Sent', 'Last Notable Activity_View in browser link Clicked', 'Page Views Per Visit', 'Lead Origin_Landing Page Submission', 
'Last Notable Activity_Email Marked Spam', 'Last Activity_Had a Phone Conversation', 'Lead Source_Google', 'Last Activity_Form Submitted on Website', 'A free copy of Mastering The Interview_Yes', 
'Last Notable Activity_Unsubscribed', 'Last Activity_Email Received', 'Last Notable Activity_Modified', 'Last Activity_SMS Sent', 'Lead Source_bing', 'Last Activity_Olark Chat Conversation']

no 
#select final_col list for X_train.columns from final_col
X_train_1= X_train[final_col]

lm = LogisticRegression(max_iter=1000, random_state=100, class_weight='balanced')
lm.fit(X_train_1, y_train)
rfe = RFE(estimator=lm, n_features_to_select=20)
rfe = rfe.fit(X_train_1, y_train)
list(zip(X_train_1.columns,rfe.support_,rfe.ranking_))

#col with rfe support true
col = X_train_1.columns[rfe.support_]
#col with rfe support false
nots = X_train_1.columns[~rfe.support_]

#print both col
print(col)
print(nots)
print(len(final_col))


# In[170]:


#model building using statsmodel
import statsmodels.api as sm
#create function
def build_model(cols):
    X_train_sm = sm.add_constant(X_train[cols])
    logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
    res = logm1.fit()
    print(res.summary())
    print("-vif-"*20)

    vif = pd.DataFrame()
    vif['Features'] = X_train[cols].columns
    vif['VIF'] = [variance_inflation_factor(X_train[cols].values, i) for i in range(X_train[cols].shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
    print("---roc---"*20)
    #predict
    y_train_pred = res.predict(X_train_sm).values.reshape(-1)
    y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
    y_train_pred_final['Lead Number'] = y_train.index
    y_train_pred_final.head()
    #create new column
    y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
    #confusion matrix
    conf = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
    draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
    #print accuracy
    print("Accuracy",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
    #print sensitivity
    print("Sensitivity",conf[1,1]/(conf[1,0]+conf[1,1]))
    #print specificity
    print("Specificity",conf[0,0]/(conf[0,0]+conf[0,1]))
    #print precision
    print("Precision",conf[1,1]/(conf[0,1]+conf[1,1]))
    #print recall
    print("Recall",conf[1,1]/(conf[1,0]+conf[1,1]))
    #print f1 score
    print("F1 Score",metrics.f1_score(y_train_pred_final.Converted, y_train_pred_final.predicted))

    return conf


#graph for ROC curve
def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate = False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(20, 10))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return None

 



# In[171]:


#model1
conf = build_model(col)
print


# In[172]:


#remove Last Notable Activity_View in browser link Clicked
col_2 = col.drop('Last Activity_Page Visited on Website', 1)
#model2
conf_2 = build_model(col_2)
print(conf_2)


# In[173]:


#drop A free copy of Mastering The Interview_Yes
col_3=col_2.drop('Last Activity_SMS Sent',1)
#model3
conf_3 = build_model(col_3)
print(conf_3)


# In[174]:


#remove Last Notable Activity_View in browser link Clicked
col_4 = col_3.drop('Last Notable Activity_View in browser link Clicked', 1)
#model4
conf_4 = build_model(col_4)
print(conf_4)


# In[175]:


#remove Last Notable Activity_Modified after removing as it means otheing hard to expain 
col_5 = col_4.drop('Last Notable Activity_Modified', 1)
#model5
conf_5 = build_model(col_5)
print(conf_5)


# In[176]:


#remove 
col_6 = col_5.drop('Last Activity_Email Opened', 1)
#model6
conf_6 = build_model(col_6)
print(conf_6)


# In[177]:


#remove 
col_7 = col_6.drop('A free copy of Mastering The Interview_Yes', 1)
#model7
conf_7 = build_model(col_7)
print(conf_7)


# In[178]:


#remove Lead Origin_Landing Page Submission
col_8 = col_7.drop('Lead Origin_Landing Page Submission', 1)
#model8
conf_8 = build_model(col_8)
print(conf_8)


# In[179]:


# remove Last Notable Activity_Email Marked Spam
col_9 = col_8.drop('Last Notable Activity_Email Marked Spam', 1)
#model9
conf_9 = build_model(col_9)
print(conf_9)


# In[180]:


#remove Last Activity_Had a Phone Conversation
col_10 = col_9.drop('Last Activity_Had a Phone Conversation', 1)

#model10
conf_10 = build_model(col_10)
print(conf_10)


# ## Finding Optimal Cutoff Point

# In[181]:


y_train_pred_final=build_model(col_10)


# In[182]:


#stat model buld in col_10
X_train_sm = sm.add_constant(X_train[col_10])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
print(res.summary())


# In[183]:


#predict
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Lead Number'] = y_train.index
y_train_pred_final.head()


# ## Finding Optimal Cutoff Point

# In[184]:


# Finding Optimal Cutoff Point
number=[float(x)/10 for x in range(10)]
for i in number:
    y_train_pred_final[i]=y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[185]:


#  calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix
num_col = [float(x)/10 for x in range(10)]
for i in num_col:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
#plot
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])


# In[186]:


#From the curve above, 0.42 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.42 else 0)
y_train_pred_final.head()



# In[187]:


#confusion matrix
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[188]:


# true positive true negative false positive false negative
TP = confusion2[1,1] # true positive
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


print(f"sensitivity: {round(TP / float(TP+FN),2)*100 }%")
print(f"specificity: {round(TN / float(TN+FP),2)*100 }%")

print(f"precision: {round(TP / float(TP+FP),2)*100 }%")
print(f"recall: {round(TP / float(TP+FN),2)*100 }%")

print(f"f1 score: {round(2*TP / float(2*TP+FP+FN),2)*100 }%")





# In[189]:


#Precision and recall tradeoff
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## Making predictions on the test set

# In[190]:


#Making predictions on the test set
X_test[col_10].head()
X_test_sm = sm.add_constant(X_test[col_10])
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Putting Prospect ID to index
y_test_df['Lead Number'] = y_test_df.index

# Removing index for both dataframes to append them side by side
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

# Let's see the head of y_pred_final
y_pred_final["final_predicted"] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()





# In[191]:


## Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)
#check confusion matrix




# In[192]:


#check the correlation matrix
col_3 = col_10.append(pd.Index(['Converted']))
# col_10 columns in dataframe x_edu_df
x_edu_df_final = x_edu_df[col_3]

#corelation matrix
corr = x_edu_df_final.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot = True,cmap="YlGnBu")
plt.show()


# # Summary

# * **Final Model Summary**:
#   * **Accuracy** : 76%
#   * **Other stats are mentioned As**
#   sensitivity: 73.0%
# specificity: 81.0%
# precision: 71.0%
# recall: 73.0%
# f1 score: 72.0%
# 
# 
# * **Top 3 Indicators are**
#   * Total Time Spent on Website
#   * Lead Source_Olark Chat 
#   * Last Notable Activity_SMS Sent 
