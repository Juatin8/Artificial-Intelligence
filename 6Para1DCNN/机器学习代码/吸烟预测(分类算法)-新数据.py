#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import pandas as pd
# df1 = pd.read_excel('a.xlsx')  # 这里输入你excel的地址
# df2 = pd.read_excel('t.xlsx')
# df3 = pd.read_excel('k.xlsx')
# df1.columns = ["Timestamp", "Accelaration-X", "Accelaration-Y", "Accelaration-Z"]  # 这里可以更改你的各列的名字，df就是你要的dataframe文件
# df2.columns = ["Timestamp", "AngularVelocity-X", "AngularVelocity-Y", "AngularVelocity-Z"]
# df3.columns = ["Timestamp","latitude", "longtitude", "altitude", "speed", "course", "hacc"]


# print(df1.head())
# df3.head()


# In[15]:


# import datetime 
# result = pd.DataFrame()
# for i in [df1,df2,df3]:
#     i['minute'] = i['Timestamp'].apply(lambda x  : datetime.datetime.strftime(x,'%Y-%m-%d %H:%M'))
#     j = i.groupby('minute').mean()
#     j.reset_index()
#     if result.empty() :
#         result = pd.concat([result,j])
#     else : 
#         result = pd.merge(result,j ,on= 'minute' , how = 'inner')
    


# In[16]:


# df1['minute'] = df1['Timestamp'].apply(lambda x : datetime.datetime.strftime(x,'%Y-%m-%d %H:%M'))
# data1 = df1.groupby('minute').mean()
# data1.reset_index()


# In[17]:


# from random import seed, randrange
# values = [randrange(2) for _ in range(20)]
# print(values)


# In[18]:


# ''.join([1,2,34,5])


# In[1]:


# 导入相应的模块
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier   # 随机森林
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 线性判别分析
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neighbors import KNeighborsClassifier   # k近邻
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.svm import SVC  # 支持向量机
from sklearn.model_selection import GridSearchCV ,cross_val_score , StratifiedKFold,KFold
from datetime import datetime 

# 导入数据

smoke = pd.read_csv('D:\modified_data.csv')
smoke = smoke.iloc[:,1:]
smoke.head()


# In[2]:


smoke.info()


# In[3]:


# 数据预处理
# 查看缺失值
# print(smoke.isnull().sum())
# # 缺失值填充 向前填充用前一个非缺失值去填充该缺失值
# smoke.fillna(method = 'ffill',inplace =True) 

# # 无法填充的行则先删除
# smoke.dropna(inplace =True)
# # 列名重命名 
# smoke.rename(columns = {'minute':'datetime'} , inplace =True)
# # lable is_smoke 
# smoke['is_smoke']  =smoke['is_smoke'].astype(int)
# # 特征工程  
# # 吸烟一般跟一天的时间有相关性，跟日期关系极小，可忽略不计。 如果要细分则工作日，节假日以及休息日
# for j,i in enumerate(['hour','minute','second'],start = 0):
#     smoke.insert(1,i,smoke['datetime'].apply(lambda x : int((x.split()[1].split(':')[j]))))


# In[4]:


# 显示有缺失值的行
smoke[smoke.isnull().sum(axis = 1)>0]


# In[5]:


# 划分训练集和测试集

train = smoke 
x = train.values[:,1:-1]
y = train.values[:,-1]


# 设置kfold,交叉采样法拆分数据集
kfold = KFold(n_splits=10,shuffle=True)  # 10折交叉

# 将分类器存入到一个列表中
classifiers = [SVC(),
               DecisionTreeClassifier(),
               RandomForestClassifier(),
              KNeighborsClassifier(),
              LogisticRegression(),
              LinearDiscriminantAnalysis()] 


# 训练模型
cv_results = []
for classifier in classifiers : 
    cv_results.append(cross_val_score(classifier , x , y.astype(int) ,scoring = 'accuracy' ,cv = kfold ,n_jobs = -1))
    
cv_results


# In[ ]:


#不同机器学习交叉验证结果汇总
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

#汇总数据
cvResDf = pd.DataFrame({'cv_mean':cv_means,'cv_std':cv_std,'algorithm':['SVC','DecisionTreeCla','RandomForestCla','KNN','LR','LinearDiscrimiAna']})
cvResDf


# In[ ]:


# 用决策树分类器进行训练
dt  = DecisionTreeClassifier(criterion='entropy')
dt.fit(x,y.astype(int)) 


# In[ ]:


#随机拿前200行进行测试
from random import seed,randrange
seed(200) # 设置随机种子
values = [randrange(len(smoke)) for _ in range(200)]
test = smoke.iloc[values,1:-1]
test.head()


# In[ ]:


# 进行预测
pred = dt.predict(test)
y = smoke.iloc[values,-1].values
pred == y


# In[ ]:


smoke.head()


# In[ ]:




