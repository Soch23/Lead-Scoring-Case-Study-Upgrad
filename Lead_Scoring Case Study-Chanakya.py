#!/usr/bin/env python
# coding: utf-8

# <b><font color= blue size =4>Lead Scoring - Case Study</font></b><br>
# <b><font color = maroon>Problem Statement</font></b><br>
# An X Education need help to select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires us to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.
# 
# <b><font color = maroon>Goals of Case Study</font></b><br>
# 
# Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.
# 

# <b><font color= blue size =4>Step 1 : Importing Libraries and Data</font></b>

# In[1]:


#Suppresssing warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing dataset to csv

leads_df=pd.read_csv("Leads.csv")


# <b><font color= blue size =4>Step 2: Inspecting the Dataframe </font></b>

# In[4]:


#Let's see the head of our dataset
leads_df.head()


# In[5]:


#Let's check the dimesions of the dataframe
leads_df.shape


# In[6]:


#Statstical aspects of the dataframe
leads_df.describe()


# In[7]:


#Let's check out info of each column
leads_df.info()


# #### Here, we can see the presence of few categorical values for which we have to create dummy variables. Also, presence of null values can be observed thus, we have to treat them accordingly in further steps

# In[8]:


#check for duplicates
sum(leads_df.duplicated(subset = 'Prospect ID')) == 0


# In[9]:


#check for duplicates
sum(leads_df.duplicated(subset = 'Lead Number')) == 0


# #### No duplicate values exist in 'Prospect ID' and 'Lead Number'

# ## <font color=purple>Exploratory Data Analysis</font>

# # <b><font color= blue size =4>Step 3: Data Cleaning</font></b>

# <b><font color= maroon size =3>3.1 Identifying Missing Values</font></b>

# In[10]:


#dropping Lead Number and Prospect ID since they have all unique values

leads_df.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# #### As can be seen,there are few columns with level called 'Select' which means that the customer had not selected the option for that particular column which is why it shows 'Select'. These values are as good as missing values and hence we will convert 'Select' values to Nan

# In[11]:


#Replacing 'Select' values with Nan
leads_df=leads_df.replace("Select", np.nan)


# In[12]:


#Checking for count of missing values in each column
leads_df.isnull().sum()


# In[13]:


#checking percentage of null values in each column

round(100*(leads_df.isnull().sum()/len(leads_df.index)), 2)


# #### As we can see there are many columns with high percentage of null values, we will drop them as they are not useful

# <b><font color= maroon size =3>3.2 Dropping Columns with Missing Values >=35%</font></b>

# In[14]:


#Drop all the columns with more than 45% missing values
cols=leads_df.columns

for i in cols:
    if((100*(leads_df[i].isnull().sum()/len(leads_df.index))) >= 35):
        leads_df.drop(i, 1, inplace = True)


# In[15]:


#checking percentage of null values in each column after dropping columns with more than 45% missing values

round(100*(leads_df.isnull().sum()/len(leads_df.index)), 2)


# <b><font color= maroon size =3>3.3 Categorical Attributes Analysis: </font></b>
#  

# <b><font color= green size =3>Imbalanced Variables</font></b>

# In[16]:


# Visualzing  variables for imbalancing
fig, axs = plt.subplots(3,4,figsize = (20,12))
sns.countplot(x = "Search", hue = "Converted", data = leads_df, ax = axs[0,0],palette = 'Set2')
sns.countplot(x = "Magazine", hue = "Converted", data = leads_df, ax = axs[0,1],palette = 'Set2')
sns.countplot(x = "Newspaper Article", hue = "Converted", data = leads_df, ax = axs[0,2],palette = 'Set2')
sns.countplot(x = "X Education Forums", hue = "Converted", data = leads_df, ax = axs[0,3],palette = 'Set2')
sns.countplot(x = "Newspaper", hue = "Converted", data = leads_df, ax = axs[1,0],palette = 'Set2')
sns.countplot(x = "Digital Advertisement", hue = "Converted", data = leads_df, ax = axs[1,1],palette = 'Set2')
sns.countplot(x = "Through Recommendations", hue = "Converted", data = leads_df, ax = axs[1,2],palette = 'Set2')
sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = leads_df, ax = axs[1,3],palette = 'Set2')
sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data =leads_df, ax = axs[2,0],palette = 'Set2')
sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = leads_df, ax = axs[2,1],palette = 'Set2')
sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = leads_df, ax = axs[2,2],palette = 'Set2')
sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = leads_df, ax = axs[2,3],palette = 'Set2')
plt.show()


# <b><font size= 3> Inference </font></b><br><ul><li> For all these columns  except 'A free copy of Mastering The Interview' data is highly imbalanced, thus we will drop them</li><li> "A free copy of Mastering The Interview" is a redundant variable so we will include this also in list of dropping columns.</li>

# In[17]:


#creating a list of columns to be dropped

cols_to_drop=(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview'])


# <b><font color= green size =3> Lead Source</font></b>

# In[18]:


#checking value counts of Lead Source column

leads_df['Lead Source'].value_counts(dropna=False)


# #### Google is having highest number of occurences, hence we will impute the missing values with label 'Google'

# In[19]:


#replacing Nan Value with Google
leads_df['Lead Source'] = leads_df['Lead Source'].replace(np.nan,'Google')

#'Lead Source' is having same label name 'Google' but in different format i.e 'google', So converting google to Google
leads_df['Lead Source'] = leads_df['Lead Source'].replace('google','Google')


# In[20]:


#combining low frequency values to Others

leads_df['Lead Source'] = leads_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM','Live Chat'] ,'Others')


# In[21]:


#visualizing count of Lead Source Variable based on Converted value
plt.figure(figsize=(15,5))
s1=sns.countplot(x= 'Lead Source', hue='Converted' , data =leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()


# <b><font size= 3> Inference </font></b><br>
# <ul><li>Maximum Leads are generated by Google and Direct Traffic.</li>
#     <li>Conversion rate of Reference leads and Welinkgak Website leads is very high.</li>
#     </ul> 

# <b><font color= green size =3> Country</font></b>

# In[22]:


#checking value counts of Country column

leads_df['Country'].value_counts(dropna=False)


# #### Since, missing values are very high , we can impute all missing values with  value 'not provided'
# 

# In[23]:


#Imputing missing values in Country column with "'not provided"
leads_df['Country'] = leads_df['Country'].replace(np.nan,'not provided')


# In[24]:


# Visualizing Country variable after imputation
plt.figure(figsize=(15,5))
s1=sns.countplot(x= 'Country', hue='Converted' , data =leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# <b><font size= 3> Inference </font></b><br>As we can see that most of the data consists of value 'India', no inference can be drawn from this parameter.Hence, we can drop this column

# In[25]:


#creating a list of columns to be droppped

cols_to_drop.append('Country')

#checking out list of columns to be dropped
cols_to_drop


# <b><font color= green size =3> What is your current occupation</font></b>

# In[26]:


#checking value counts of 'What is your current occupation' column
leads_df['What is your current occupation'].value_counts(dropna=False)


# ####  Since no information has been provided regarding occupation, we can replace missing values with new category 'Not provided' 
# 

# In[27]:


#Creating new category 'Not provided'

leads_df['What is your current occupation'] = leads_df['What is your current occupation'].replace(np.nan, 'Not provided')


# In[28]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(x='What is your current occupation', hue='Converted' , data = leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()


# <b><font size= 3> Inference </font></b><br>
# <ul><li>Maximum leads generated are unemployed and their conversion rate is more than 50%.</li>
# <li>Conversion rate of working professionals is very high.</li></ul>

# <b><font color= green size =3> What matters most to you in choosing a course</font></b>

# In[29]:


#checking value counts of 'What matters most to you in choosing a course'

leads_df['What matters most to you in choosing a course'].value_counts(dropna=False)


# #### Clearly seen that missing values in the this column can be imputed by 'Better Career Prospects'

# In[30]:


leads_df['What matters most to you in choosing a course'] = leads_df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[31]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(x= 'What matters most to you in choosing a course', hue='Converted' , data = leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()


# <b><font size= 3> Inference </font></b><br> This column spread of variance is very low , hence it can be dropped.
# 

# In[32]:


# Append 'What matters most to you in choosing a course'to the cols_to_drop List
cols_to_drop.append('What matters most to you in choosing a course')

#checking updated list for columns to be dropped
cols_to_drop


# <b><font color= green size =3>Last Activity</font></b>

# In[33]:


#checking value counts of Last Activity
leads_df['Last Activity'].value_counts(dropna=False)


# #### Missing values can be imputed with mode value "Email Opened"

# In[34]:


#replacing Nan Values with mode value "Email Opened"

leads_df['Last Activity'] = leads_df['Last Activity'].replace(np.nan,'Email Opened')


# In[35]:


#combining low frequency values
leads_df['Last Activity'] = leads_df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                       'Had a Phone Conversation', 
                                                       'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[36]:


#visualizing count of Last Activity Variable 

plt.figure(figsize=(15,5))
s1=sns.countplot(x='Last Activity', hue='Converted' , data = leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# <b><font size= 3> Inference </font></b><br><ul><li> Maximum leads are generated having last activity as Email opened but conversion rate is not too good.</li>
#     <li> SMS sent as last acitivity has high conversion rate.</li> 

# In[37]:


# Append 'Last Activity' to the cols_to_drop List it is a X-education's sales team generated data
cols_to_drop.append('Last Activity')

#checking updated list for columns to be dropped
cols_to_drop


# In[38]:


#Check the Null Values in All Columns after imputation:
round(100*(leads_df.isnull().sum()/len(leads_df.index)), 2)


# In[39]:


# Remaining missing values percentage is less than 2%, we can drop those rows without affecting the data
leads_df = leads_df.dropna()


# In[40]:


leads_df.shape


# In[41]:


#Checking percentage of Null Values in All Columns:
round(100*(leads_df.isnull().sum()/len(leads_df.index)), 2)


# <b><font color= green size =3>Lead Origin</font></b>

# In[42]:


s1=sns.countplot(x='Lead Origin', hue='Converted' , data = leads_df , palette = 'Set2')
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# <b><font color= green size =3>Do Not Email & Do Not Call</font></b>

# In[43]:


fig, axs = plt.subplots(1,2,figsize = (15,7.5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = leads_df, ax = axs[0],palette = 'Set2')
sns.countplot(x = "Do Not Call", hue = "Converted", data = leads_df, ax = axs[1],palette = 'Set2')
plt.show()


# #### We Can append the Do Not Call Column to the list of Columns to be Dropped data is higjly skewed

# In[44]:


# Append 'Do Not Call' to the cols_to_drop List
cols_to_drop.append('Do Not Call')

#checking updated list for columns to be dropped
cols_to_drop


# <b><font color= green size =3>Last Notable Activity</font></b>

# In[45]:


#checking value counts of last Notable Activity
leads_df['Last Notable Activity'].value_counts()


# In[46]:


#clubbing lower frequency values

leads_df['Last Notable Activity'] = leads_df['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Others')


# In[47]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = leads_df , palette = 'Set2')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# <b><font size= 3> Inference </font></b><br><ul><li> Maximum leads are generated having last activity as Email opened but conversion rate is not too good.</li>
#     <li> SMS sent as last acitivity has high conversion rate.</li> 

# In[48]:


# Append 'Last Notable Activity'to the cols_to_drop List as this is a sales team generated data
cols_to_drop.append('Last Notable Activity')


# In[49]:


# checking final list of columns to be dropped
cols_to_drop


# In[50]:


#dropping columns
leads = leads_df.drop(cols_to_drop,1)

#checking info of dataset for remaining columns
leads.info()


# In[51]:


#checking dataset
leads.head()


# 

# ### <font color =blue>3.4 Numerical Attributes Analysis:</font>

# <b><font color= green size =3>Converted</font></b>

# In[52]:


#Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).
#Visualizing Distribution of 'Converted' Variable
sns.countplot(leads.Converted)
plt.xlabel("Converted Status")
plt.ylabel("Count of Target")
plt.title("Distribution of 'Converted' Variable")
plt.show()


# In[53]:


# Finding out conversion rate
Converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100
Converted


# #### Currently, lead Conversion rate is 38% only 

# In[54]:


#Checking correlations of numeric values using heatmap

# Size of the figure
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(leads.corr(), cmap="YlGnBu", annot=True)
plt.show()


# <b><font color= green size =3>Total Visits</font></b>

# In[55]:


#visualizing spread of variable Total Visits

sns.boxplot(y=leads['TotalVisits'])
plt.show()


# #### Presence of outliers can be seen clearly

# In[56]:


#checking percentile values for "Total Visits"

leads['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[57]:


#Outlier Treatment: capping the outliers to 95% value for analysis

percentiles = leads['TotalVisits'].quantile([0.05,0.95]).values
leads['TotalVisits'][leads['TotalVisits'] <= percentiles[0]] = percentiles[0]
leads['TotalVisits'][leads['TotalVisits'] >= percentiles[1]] = percentiles[1]

#visualizing variable after outlier treatment
sns.boxplot(y=leads['TotalVisits'])
plt.show()


# In[58]:


# Visualizing TotalVisits w.r.t Target Variable 'Converted'
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leads)
plt.show()


# <b><font size= 3> Inference </font></b><br> As the median for both converted and non-converted leads are same , nothing coclusive can be said on the basis of variable TotalVisits

# <b><font color= green size =3>Total time spent on website</font></b>

# In[59]:


#checking percentiles for "Total Time Spent on Website"

leads['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[60]:


#visualizing spread of variable 'Total Time Spent on Website'
sns.boxplot(y = leads['Total Time Spent on Website'])
plt.show()


# #### Since there are no major outliers for the above variable, outlier treatment is not required for it

# In[61]:


# Visualizing 'Total Time Spent on Website' w.r.t Target Variable 'converted'
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = leads)
plt.show()


# <b><font size= 3> Inference </font></b><br> As can be seen, leads spending more time on website are more likely to convert , thus website should be made more enagaging to increase conversion rate

# <b><font color= green size =3>Page views per visit</font></b>

# In[62]:


leads['Page Views Per Visit'].describe()


# In[63]:


#visualizing spread of variable 'Page Views Per Visit'
sns.boxplot(y =leads['Page Views Per Visit'])
plt.show()


# #### Presence of outliers can be clearly seen in the above boxplot, thus outlier treatment need to be done for this variable

# In[64]:


#Outlier Treatment: capping the outliers to 95% value for analysis
percentiles = leads['Page Views Per Visit'].quantile([0.05,0.95]).values
leads['Page Views Per Visit'][leads['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
leads['Page Views Per Visit'][leads['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]

#visualizing variable after outlier treatment
sns.boxplot(y=leads['Page Views Per Visit'])
plt.show()


# In[65]:


#visualizing 'Page Views Per Visit' w.r.t Target variable 'Converted'
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = leads)
plt.show()


# <b><font size= 3>Inference</font></b><br> 
# 
# <ul><li>Median for converted and not converted leads is almost same.</li>
# <li>Nothing conclusive can be said on the basis of Page Views Per Visit.</li></ul>

# In[66]:


# Now check the conversions for all numeric values

plt.figure(figsize=(20,20))
plt.subplot(4,3,1)
sns.barplot(y = 'TotalVisits', x='Converted', palette='Set2', data = leads)
plt.subplot(4,3,2)
sns.barplot(y = 'Total Time Spent on Website', x='Converted', palette='Set2', data = leads)
plt.subplot(4,3,3)
sns.barplot(y = 'Page Views Per Visit', x='Converted', palette='Set2', data = leads)
plt.show()


# 
# <b><font size= 3>Inference</font></b><br> 
# The conversion rate is high for Total Visits, Total Time Spent on Website and Page Views Per Visit

# ### Now, all data labels are in good shape , we will proceed to our next step which is Data Preparation

# # <b><font color= blue size =4>Step 4: Data Preparation</font></b>

# <b><font color= maroon size =3>4.1 Converting some binary variables (Yes/No) to 0/1</font></b>

# In[67]:


# List of variables to map

varlist =  ['Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
leads[varlist] = leads[varlist].apply(binary_map)


# In[68]:


leads.head()


# In[69]:


leads.info()


# <b><font color= maroon size =3>4.2 Dummy Variable Creation:</font></b>

# In[70]:


#getting a list of categorical columns foy creating dummy

cat_cols= leads.select_dtypes(include=['object']).columns
cat_cols


# In[71]:


#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(leads[['Lead Origin']], drop_first=True)
leads = pd.concat([leads,dummy],1)



dummy = pd.get_dummies(leads['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
leads = pd.concat([leads, dummy], axis = 1)


dummy = pd.get_dummies(leads['What is your current occupation'], prefix  = 'What is your current occupation')
dummy = dummy.drop(['What is your current occupation_Not provided'], 1)
leads = pd.concat([leads, dummy], axis = 1)



# In[72]:


#dropping the original columns after dummy variable creation

leads.drop(cat_cols,1,inplace = True)


# In[73]:


#checking dataset after dummy variable creation
leads.head()


# # <b><font color= blue size =4>Step 5: Test-Train Split</font></b>

# In[74]:


#importing library for splitting dataset
from sklearn.model_selection import train_test_split


# In[75]:


# Putting feature variable to X
X=leads.drop('Converted', axis=1)

#checking head of X
X.head()


# In[76]:


# Putting response variable to y
y = leads['Converted']

#checking head of y
y.head()


# In[77]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# # <b><font color= blue size =4>Step 6: Feature Scaling</font></b>

# In[78]:


#importing library for feature scaling
from sklearn.preprocessing import StandardScaler


# In[79]:


#scaling of features
scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

#checking X-train dataset after scaling
X_train.head()


# In[80]:


## Checking the conversion rate from 'converted' column as it denotes the target variable

(sum(y)/len(y.index))*100


# In[81]:


# Let's see the correlation matrix
plt.figure(figsize = (20,15))        # Size of the figure
sns.heatmap(leads.corr(),annot = True)
plt.show()


# <b><font color= maroon size =3>Dropping highly correlated dummy variables</font><b>

# In[82]:


X_test = X_test.drop(['Lead Source_Olark Chat','Lead Origin_Landing Page Submission'],1)


# In[83]:


X_train = X_train.drop(['Lead Source_Olark Chat','Lead Origin_Landing Page Submission'],1)


# # <b><font color= blue size =4>Step 7: Model Building using Stats Model & RFE</font></b>

# In[84]:


# importing necessary library
import statsmodels.api as sm


# In[85]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[86]:


rfe.support_


# In[87]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[88]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[89]:


X_train.columns[~rfe.support_]


# <b><font color= green size =3>Model 1</font></b>

# In[90]:


#BUILDING MODEL #1

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# #### p-value of variable What is your current occupation_Housewife is high, so we can drop it.

# In[91]:


#dropping column with high p-value

col = col.drop('What is your current occupation_Housewife',1)


# <b><font color= green size =3>Model 2</font></b>

# In[92]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# <b>p-value of variable "Lead Source_Welingak Website" is high, so we will drop it.

# In[93]:


#dropping column with high p-value

col = col.drop('Lead Source_Welingak Website',1)


# <b><font color= green size =3>Model 3</font></b>

# In[94]:


#BUILDING MODEL #3

X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# <b>variable 'What is your current occupation_Businessman' has high p-value, so it needs to be dropped

# In[95]:


#dropping column with high p-value

col = col.drop('What is your current occupation_Businessman',1)


# <b><font color= green size =3>Model 4</font></b>

# In[96]:


#BUILDING MODEL #4

X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[97]:


#dropping column with high p-value

col = col.drop('What is your current occupation_Other',1)


# <b><font color= green size =3>Model 5</font></b>

# In[98]:


#BUILDING MODEL #5

X_train_sm = sm.add_constant(X_train[col])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()


# <b><font color = maroon>Since the Model 5 seems to be stable with significant p-values, we shall go ahead with this model for further analysis

# <b><font color= green size =3>Calculating VIF</font></b>

# In[99]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# <b>All variables have a good value of VIF. So we need not drop any more variables and we can proceed with making predictions using this model only

# ## <font color =maroon>Predicting a Train model</font>

# In[100]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[101]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[102]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[103]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# <b><font color =green size =3>Metrics -Accuracy, Sensitivity, Specificity, False Positive Rate, Postitive Predictive Value and Negative Predictive Value</font>

# In[104]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[105]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[106]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[107]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[108]:


# Let us calculate specificity
TN / float(TN+FP)


# In[109]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[110]:


# positive predictive value 
print (TP / float(TP+FP))


# In[111]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### <font color = maroon>PLOTTING ROC CURVE

# An ROC curve demonstrates several things:<br>
# 
# <ul><li>It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).</li>
# <li>The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.</li>
# <li>The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.</li></ul>

# In[112]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[113]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[114]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# <B>The ROC Curve should be a value close to 1. We are getting a good value of 0.86 indicating a good predictive model.</B>

# ### <font color = maroon>Finding Optimal Cutoff Point

# Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[115]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[116]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[117]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# #### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

# In[118]:


y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[119]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[120]:


# checking if 80% cases are correctly predicted based on the converted column.

# get the total of final predicted conversion / non conversion counts from the actual converted rates

checking_df = y_train_pred_final.loc[y_train_pred_final['Converted']==1,['Converted','final_Predicted']]
checking_df['final_Predicted'].value_counts()


# In[121]:


# check the precentage of final_predicted conversions

2005/float(2005+414)


# ### <font color = maroon>Hence, we can see that the final prediction of conversions have a target of 83% conversion as per the X Educations CEO's requirement . Hence, we can say that this is a good model.

# <b><font color = green size =3>Overall Metrics - Accuracy, Confusion Metrics, Sensitivity, Specificity, False Postive Rate, Positive Predictive Value, Negative Predicitive Value on final prediction on train set

# In[122]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[123]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[124]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[125]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[126]:


# Let us calculate specificity
TN / float(TN+FP)


# <b><font size=3>Inference:</font></b><br>
# So as we can see above the model seems to be performing well. The ROC curve has a value of 0.86, which is very good. We have the following values for the Train Data:
# <ul>
#     <li>Accuracy : 77.05%</li>
#     <li>Sensitivity :82.89%</li>
#     <li>Specificity : 73.49%</li></ul>
#     
# Some of the other Stats are derived below, indicating the False Positive Rate, Positive Predictive Value,Negative Predictive Values, Precision & Recall.

# In[127]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[128]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[129]:


# Negative predictive value
print (TN / float(TN+ FN))


# <b><font color= green size =3>Precision and Recall</font></b>

# In[130]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[131]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[132]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[133]:


from sklearn.metrics import precision_score, recall_score


# In[134]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[135]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# <b><font color= green size =3>Precision and Recall Trade-off</font></b>

# In[136]:


# importing precision recall curve from sklearn library
from sklearn.metrics import precision_recall_curve


# In[137]:


# Creating precision recall curve
y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[138]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ## <font color =maroon>Predictions on  the test set

# In[139]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[140]:


X_test = X_test[col]
X_test.head()


# In[141]:


X_test_sm = sm.add_constant(X_test)


# In[142]:


X_test_sm.shape


# In[143]:


y_test_pred = res.predict(X_test_sm)    


# In[144]:


y_test_pred[:10]


# In[145]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[146]:


# Let's see the head
y_pred_1.head()


# In[147]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[148]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[149]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[150]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[151]:


y_pred_final.head()


# In[152]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[153]:


y_pred_final.head()


# #### <font color = green>Assigning Lead Score

# In[154]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[155]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[156]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[157]:


y_pred_final.head()


# In[158]:


# checking if 80% cases are correctly predicted based on the converted column.

# get the total of final predicted conversion or non conversion counts from the actual converted rates

checking_test_df = y_pred_final.loc[y_pred_final['Converted']==1,['Converted','final_Predicted']]
checking_test_df['final_Predicted'].value_counts()


# In[159]:


# check the precentage of final_predicted conversions on test data

865/float(865+177)


# ### <font color = maroon>Hence we can see that the final prediction of conversions have a target rate of 83%  (same as predictions made on training data set)

# <b><font color = green size =3>Overall Metrics - Accuracy, Confusion Metrics, Sensitivity, Specificity, False Postive Rate, Positive Predictive Value, Negative Predicitive Value on final prediction on test set

# In[160]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[161]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[162]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[163]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[164]:


# Let us calculate specificity
TN / float(TN+FP)


# <b><font color = green size =3>Precision and Recall metrics for the test set

# In[165]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[166]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# <b><font size=3>Inference:</font></b><br>
# After running the model on the Test Data these are the figures we obtain:
# <ul>
# <li>Accuracy : 77.52%</li>
# <li>Sensitivity :83.01%</li>
# <li>Specificity : 74.13%</li>

# <b><font size =4>Conclusion:</font></b>
# 
# 
# - While we have checked both Sensitivity-Specificity as well as Precision and Recall Metrics, we have considered the
#   optimal 
#   cut off based on Sensitivity and Specificity for calculating the final prediction.
# - Accuracy, Sensitivity and Specificity values of test set are around 77%, 83% and 74% which are approximately closer to 
#   the respective values calculated using trained set.
# - Also the lead score calculated in the trained set of data shows the conversion rate on the final predicted model is 
#   around 80%
# - Hence overall this model seems to be good.   
# 
# <b>Important features responsible for good conversion rate or the ones' which contributes more towards the probability of a lead getting converted are :</b>
# <li>Lead Origin_Lead Add Form</li>
# <li>What is your current occupation_Working Professional</li>
# <li>Total Time Spent on Website	</li>

# In[ ]:




