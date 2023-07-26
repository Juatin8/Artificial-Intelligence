import torch
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

#--------------------------------- 归一化处理--------------------------------------
def Nomarlization(X):
    scaler = MinMaxScaler() # 创建MinMaxScaler缩放器
    X_scaled = scaler.fit_transform(X) # 对特征矩阵进行缩放
    return X_scaled
##### -------------------- 模型检验 -----------------------------------
def validation(y_test, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 计算精度、召回率和 F1 值
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}")
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm
def plot_cm(cm):
    # 可视化混淆矩阵
    sns.set(font_scale=1.4) # 设置字体大小
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues') # 设置参数
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')
    plt.show()
#------------------------------ LogisticRegression------------------------------
def lr_multiclass(X,y):
    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建逻辑回归模型
    clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
    # 拟合模型
    clf.fit(X_train, y_train)
    # 预测测试集
    y_pred = clf.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1

#-------------------------------- KNN-------------------------------------------------
def KNN(X,y,k):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建KNN分类器，指定k值为5
    knn = KNeighborsClassifier(n_neighbors=k)
    # 训练KNN分类器
    knn.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = knn.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1


#--------------------------------- 朴素贝叶斯------------------------------------------
def NB_multiclass(X,y):
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建多项式朴素贝叶斯分类器
    nb = MultinomialNB()
    # 训练朴素贝叶斯分类器
    nb.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = nb.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1

#------------------------------支持向量机-------------------------------------
def SVM(X,y):
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建支持向量机分类器
    svc = SVC(kernel='linear', C=1, decision_function_shape='ovr')
    # 训练支持向量机分类器
    svc.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = svc.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1


#---------------------------- 多层感知机---------------------------------
def MLP(X,y,hl_sizes,max_iter_num):
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建多层感知机模型
    mlp = MLPClassifier(hidden_layer_sizes=hl_sizes, max_iter=max_iter_num)
    # 训练模型
    mlp.fit(X_train, y_train)
    # 预测测试集的标签
    y_pred = mlp.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1
    

#------------------------- 随机森林-------------------------------------
def Random_forest(X,y,tree_num, depth_num):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建随机森林分类器对象
    clf = RandomForestClassifier(n_estimators=tree_num, max_depth=depth_num, random_state=42)
    # 在训练集上训练分类器
    clf.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = clf.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1


#--------------------------- ---决策树------------------------------------
def Decision_tree(X,y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建Decision Tree分类器对象
    clf = DecisionTreeClassifier()
    # 在训练集上训练分类器
    clf.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = clf.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1


#--------------------------- 线性判别分析-----------------------
def LDA(X,y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建LDA对象
    lda = LinearDiscriminantAnalysis()
    # 在训练集上拟合LDA模型
    lda.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = lda.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1


# --------------------------- gradient boosting----------------------------
def Gradient_boosting(X,y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建Gradient Boosting分类器对象
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
    # 在训练集上训练分类器
    clf.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = clf.predict(X_test)
    # 模型检验
    accuracy, precision, recall, f1, cm = validation(y_test, y_pred)
    # 可视化混淆矩阵
    plot_cm(cm)
    return accuracy, precision, recall, f1

#------------------------------ kmeans------------------------------------------
def kmeans(x: Tensor, num_clusters: int, num_iterations: int) -> Tensor:
    """PyTorch implementation of Kmeans clustering algorithm.
    
    Args:
        x (Tensor): The data tensor of shape (n_samples, n_features).
        num_clusters (int): The number of clusters.
        num_iterations (int): The number of iterations to run Kmeans.
        
    Returns:
        A tensor of shape (n_samples,) containing the cluster assignments for each data point.
    """
    n_samples, n_features = x.shape
    
    # Initialize cluster centers randomly
    centroids = x[torch.randperm(n_samples)[:num_clusters]]
    
    for i in range(num_iterations):
        # Calculate distances between data points and centroids
        distances = torch.cdist(x, centroids)
        
        # Assign data points to nearest cluster
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # Update cluster centers
        for j in range(num_clusters):
            centroids[j] = x[cluster_assignments == j].mean(dim=0)
    
    return cluster_assignments
