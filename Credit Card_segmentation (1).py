#!/usr/bin/env python
# coding: utf-8

# # Credit Card segmentation

# ### loading libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


#setting work directory
os.chdir("C:\\Users\\Monali\\OneDrive\\Desktop\\Case Study-DS\\Credit Card segmentation")


# In[3]:


os.getcwd()


# In[4]:


Data = pd.read_csv('credit-card-data.csv')


# ## Data Inspection

# In[5]:


Data.head()


# In[6]:


Data.shape


# In[7]:


Data.describe()


# Summary of this section
# 
# -  There are 8950 registers with 18 features. 
# -  The data is in numerical form, except for the #customer id (CUST_ID) which is an object containing letters and numbers. 
# -  On average, clients maintain 1564 dollars in the bank account for use with the debit card. -On average, clients spend 1000      USD on purchases. About the purchase mode, on average clients spend 592 dollars on one-off purchases and 411 dollars on      
#    purchases with installments. 
# -  Good news for the bank: clients, on average, use 978 dollars as cash advancement. One must have in mind that, in general, the    taxes for cash advancement are higher than the credit card taxes. In regards to frequency, clients more frequently make       
#    purchases with installents (mean = 0.364) than one-off (mean = 0.202).
# -  Regarding credit limits on the credit card, the maximum limit is 30,000 dollars with the minimum being 50 dollars. On      
#    average, clients have a credit card limit of 4494 dollar.

# In[8]:


Data.info()


# In[9]:


# Let's see if we have duplicate entries in the data
Data.duplicated().sum()


# In[10]:


sns.pairplot(Data)


# In[11]:


plt.rcParams.update({'font.size': 12})
sns.set_style("whitegrid")
Data.hist(bins=40, figsize=(30, 30));


# We can extract some insights Ffor some of the most relevant variables:
# 
# - BALANCE left in the account is more frequent around 1000 dollars.
# - PURCHASES values concentrate below 5000 dollars.
# - BALANCE FREQUENCY - we can see that clients frequently update the balance in their accounts.
# - ONEOFF_PURCHASES and INSTALLMENT_PURCHASES - looking at the scale of the graph we notice that purchases with installments are   more frequent for values no greater than 5000 dollars and one-off purchases are more frenquent for values no greater than  
#   10000 dollars.
# - PURCHASE FREQUENCY show a segumentation of clients: one group make purchases very frequently, while the other group rarely  
#   make purchases.
# - MINIMUM PAYMENTS and PRC FULL PAYMENT - these variables show us that many clients opt for paying the minumum of their credit  
#   card bill. Very few clients pay the full bill. This is also good for the bank as taxes are high for credit card bills.
# - TENURE shows that most of the clients are long term clients (more than 12 years)

# ## Data Preprocessing

# ##### Missing value analysis

# In[12]:


#checking the missing values in data
Data.isnull().sum()


# In[13]:


#Visualization of missing values with heatmap
sns.heatmap(Data.isnull(),yticklabels = False,cbar = True, cmap = "Blues",linecolor = "Black")


# In[14]:


#As Cust_id contributes no important information we can drop it
Data.drop(['CUST_ID'], axis=1, inplace=True)


# In[15]:


#Only 1 missing value is there in CREDIT_LIMIT, which can be deleted 
Data.dropna(subset=['CREDIT_LIMIT'], inplace=True)


# In[16]:


#In MINIMUM_PAYMENTS 313 missing vales are present
#Finding out the distribution of MISSING_VALUE before imputing it with central tendency
sns.distplot(Data['MINIMUM_PAYMENTS'])


# The "MINIMUM_PAYMENT" distrbution is left screwed thus we will impute the missing values with median

# In[17]:


Data['MINIMUM_PAYMENTS'].fillna(Data['MINIMUM_PAYMENTS'].median(), inplace = True)


# In[18]:


#Checking whethera ll the missing values are handled
Data.isnull().sum().sum()


# ### Outlier Analysis

# In[19]:


graph_by_variables = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']


# In[20]:


plt.figure(figsize=(20,35))

for i in range(0,17):
    plt.subplot(6, 3, i+1)
    plt.boxplot(Data[graph_by_variables[i]].dropna())
    plt.title(graph_by_variables[i])


# In[21]:


plt.figure(figsize=(20,10))
for j in list(Data.columns.values):
    plt.scatter(y=Data[j],x=[i for i in range(len(Data[j]))],s=[20])
plt.legend()


# The separate cluster can be made from the outliers thus we won't handle it. The purchase for specific customer might be so height, it can't be considered as outlier as it may contain the useful information

# ## Finding Co-relation

# In[22]:


#Heatmap with the corellation matrix
plt.matshow(Data.corr())


# In[23]:


f, ax = plt.subplots(figsize = (20,10))
sns.heatmap(Data.corr(),annot = True)


# Correlation is stronger as the values approach 1. From the correlation matrix we take that:
# 
# PURCHASE INSTALLMENTS FREQUENCY is somehow correlated to PURCHASES FREQUENCY, and this confirms the insight. PURCHASE and ONEOFF PURCHASE are strongly correlated and it seems that most of the purchases values are related to one-off purchases. When we look at INSTALLMENTS PURCHASES correlation with PURCHASES we see that the value is 0.68, not as strong as the correlation with one-off purchases.

# ## Deriving KPI's

# In[24]:


# Monthly average purchase calculation
Data_copy = Data.copy()
Data_copy['Monthly_avg_purchase']=Data['PURCHASES']/Data['TENURE']


# In[25]:


Data_copy['Monthly_avg_purchase'].head()


# In[26]:


#Cash Advance Amount
Data_copy['Cash_advance_amount'] = Data['CASH_ADVANCE']/Data['TENURE']


# In[27]:


Data_copy['Cash_advance_amount'].head()


# In[28]:


Data.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]


# In[29]:


def purchase(Data_copy):    
    if (Data_copy['ONEOFF_PURCHASES']==0) & (Data_copy['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (Data_copy['ONEOFF_PURCHASES']>0) & (Data_copy['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (Data_copy['ONEOFF_PURCHASES']>0) & (Data_copy['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (Data_copy['ONEOFF_PURCHASES']==0) & (Data_copy['INSTALLMENTS_PURCHASES']>0):
        return 'installment'


# In[30]:


# Purchase type
Data_copy['Purchase_Type'] = Data_copy.apply(purchase, axis = 1)


# In[31]:


Data_copy['Purchase_Type'].head()


# In[32]:


# Label Encoding to convert it into numerical data
Data_copy['Purchase_Type'].replace({"none":0, "one_off": 1, "installment":2, "both_oneoff_installment":3}, inplace = True)


# In[33]:


#finding limit usage
Data_copy['limit_usage'] = Data_copy.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)


# In[34]:


Data_copy['limit_usage'].head()


# In[35]:


#payment to minimum payment ratio
Data_copy['Payment_MinPayment_ratio'] = Data_copy.apply(lambda x: x['PAYMENTS']/x['MINIMUM_PAYMENTS'], axis=1)


# In[36]:


Data_copy['Payment_MinPayment_ratio'].head()


# ### Insights on derived KPI's

# In[37]:


Data_copy.groupby('Purchase_Type').apply(lambda x: np.mean(x['limit_usage'])).plot.barh()
plt.xlabel('limit_usage')
plt.title('Purchase_Type Vs lmit_usage')


# Thus, the customers who do not do one_off and installment more likely to take cash advance

# In[38]:


Data_copy.groupby('Purchase_Type').apply(lambda x: np.mean(x['Cash_advance_amount'])).plot.barh()
plt.title('Average cash advance taken by customers of different Purchase type : Both, None,Installment,One_Off')
plt.xlabel('Cash_advance_amount')


# Customers with installment purchases are paying dues

# In[39]:


x = Data_copy.groupby('Purchase_Type').apply(lambda x: np.mean(x['Payment_MinPayment_ratio']))
type(x)
x.values
fig,ax=plt.subplots()
ax.barh(y=range(len(x)), width=x.values,align='center')
ax.set(yticks= np.arange(len(x)),yticklabels = x.index);
plt.title('Mean payment_minpayment ratio for each purchse type')
plt.xlabel('Purchase_Type')
plt.ylabel('Payment_MinPayment_ratio')


# Average payment_minpayment ratio for each purchse type

# ## Profiling

# In[40]:


get_ipython().system('pip install pandas-profiling')


# In[41]:


import pandas_profiling
from pandas_profiling import ProfileReport


# In[42]:


profile = ProfileReport(Data_copy, title="Pandas Profiling Report")


# In[60]:


profile


# ## Standardization

# In[43]:


# Let's scale the data first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[44]:


scaled_data = scaler.fit_transform(Data)
scaled_data.shape


# ### Finding optimal number of cluster using Elbow method

# In[45]:


from sklearn.cluster import KMeans
scores_1 = []

range_values = range(1,20)
for i in range_values:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    scores_1.append(kmeans.inertia_)
plt.plot(scores_1, 'bx-')
plt.style.use('ggplot')
plt.title('Finding the right number of clusters')
plt.xlabel('Clusters')
plt.ylabel('Scores') 
plt.show()


# In[46]:


# From this we can observe that, 4th cluster seems to be forming the elbow of the curve. 
# However, the values does not reduce linearly until 8th cluster. 
# Let's choose the number of clusters to be 8.


# ## Applying K-means method

# In[47]:


kmeans = KMeans(8)
kmeans.fit(scaled_data)
labels = kmeans.labels_


# In[48]:


kmeans.cluster_centers_.shape


# In[49]:


cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_,columns = [Data.columns])
cluster_centers


# In[50]:


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers,columns = [Data.columns])
cluster_centers


# In[ ]:



6th Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%

0th customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)

7th customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits

4th customer cluster (low tenure): these are customers with low tenure (7 years), low balance 

3rd customer cluster:People with average to high credit limit who make all types of purchases.

2nd customer cluster (One and gone) : This group has more people with due payments, thet have very less purchase mostkt does cash advance.
    
1st customer cluster (Basic Splitters) : Usally does payments on in installment having average credit limit.   
    
5th customer cluster (One offpurchaser) : These grop does onr-off purchases more having average credit limit.


# In[51]:


labels.shape # Labels associated to each data point


# In[52]:


y_kmeans = kmeans.fit_predict(scaled_data)
y_kmeans


# In[53]:


# concatenate the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([Data, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()


# In[54]:


# Plot the histogram of various clusters
for i in Data.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))


# In[55]:


# Obtain the principal components 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(scaled_data)
principal_comp


# In[56]:


# Create a dataframe with the two components
pca_df = pd.DataFrame(data=principal_comp,columns=['pca1','pca2'])
pca_df.sample(5)


# In[57]:


# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()


# In[58]:


plt.figure(figsize=(10,10))
plt.style.use('ggplot')
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','violet','black','gray','yellow','orange'])
plt.show()


# In[59]:


from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=8, random_state=21)
kmeans.fit(scaled_data)

print('Silhoutte score of our model is ' + str(silhouette_score(scaled_data, kmeans.labels_)))


# The Silhoutte score lies between -1 to 1
# The score of out model is 0.22, which means the clusters are distinc though there might be some overlapping.

# ### Conclusion : 

# The 8 distinct clusters are formed possesing different patterns of credit card usage.
# Thus, business can prepare unique strategies for different cluster, When the plan/strategies are more target specific it yields high profit. 
