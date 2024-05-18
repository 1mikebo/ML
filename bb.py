# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 加载数据集
data = pd.read_csv("student-mat.csv")
# 数据维度
raws, cols = data.shape
print(f"该数据集有 {raws} 个样本")
print(f"该数据集有 {cols - 1} 个特征")

# 探索性数据分析（EDA）
print("数据集的前几行：")
print(data.head())

print("\n数据集的基本信息：")
print(data.info())

print("\n数据集的描述统计信息：")
print(data.describe())

# 确定离散特征和数值特征
discrete_features = data.select_dtypes(include=['object']).columns
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

# 打印离散特征和数值特征
print("\n离散特征：", discrete_features)
print("\n数值特征：", numerical_features)

# 对离散特征进行独热编码
data = pd.get_dummies(data, columns=discrete_features)


# 将离散的数学成绩（G1, G2, G3）转化为有序类别
def convert_to_continuous(x):
    x = int(x)
    if x < 5:
        return 1  # 差
    elif 5 <= x < 10:
        return 2  # 中
    elif 10 <= x < 15:
        return 3  # 好
    else:
        return 4  # 优


data['G1'] = data['G1'].apply(convert_to_continuous)
data['G2'] = data['G2'].apply(convert_to_continuous)
data['G3'] = data['G3'].apply(convert_to_continuous)

# 确保数据集中只包含数值型特征
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns


# 替换离群值
def replace_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))


# 替换离群值并绘制箱型图
data_no_outliers = data[numerical_features].apply(replace_outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_no_outliers)
plt.title("Boxplot of Student Performance Features without Outliers")
plt.xticks(rotation=45)
plt.show()

# 计算特征与目标变量（G3）之间的相关性
correlation = data.corr()['G3'].drop('G3')

# 排序特征重要性
sorted_features = correlation.abs().sort_values(ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_features.values, y=sorted_features.index, hue=sorted_features.index, palette="viridis",
            dodge=False)
plt.title("Feature Importance for G3")
plt.xlabel("Absolute Correlation")
plt.ylabel("Features")
plt.show()

# 相关性前四的特征
top_features = sorted_features.head(4).index

# 选取特征与目标变量（G3）之间的相关性
top_correlation = data[top_features].corr()

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(top_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Top 4 Features with G3')
plt.show()

# 特征选择：选择相关性较高的特征，即前20个特征
correlation = data.corr()['G3'].drop('G3')
sorted_features = correlation.abs().sort_values(ascending=False)
top_features = sorted_features.head(20).index

# 选择特征和目标变量
X = data[top_features]
y = data['G3']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 查看划分后的训练集和测试集大小
print("\n训练集大小:", X_train.shape, y_train.shape)
print("测试集大小:", X_test.shape, y_test.shape)

# 定义模型
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Machine": SVR()
}

# 记录调优前的性能指标
pre_optimization_scores = {}

# 调优前模型训练与评估
plt.figure(figsize=(10, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_pre = model.predict(X_test)
    mse_pre = mean_squared_error(y_test, y_pred_pre)
    r2_pre = r2_score(y_test, y_pred_pre)
    mae_pre = mean_absolute_error(y_test, y_pred_pre)
    pre_optimization_scores[name] = {"MSE": mse_pre, "R2": r2_pre, "MAE": mae_pre}
    plt.bar(name, r2_pre, color='red', alpha=0.5)
plt.title("Pre-Optimization Model Performance (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.show()

# 输出每个模型的调优前性能指标
for name, scores in pre_optimization_scores.items():
    print(f"Pre-Optimization Scores for {name}:")
    print(f"MSE: {scores['MSE']}, R2: {scores['R2']}, MAE: {scores['MAE']}")

# 模型参数调优
best_models = {}
for name, model in models.items():
    if name == "Linear Regression":
        best_model = model  # 线性回归模型不需要调优
    elif name == "Decision Tree":
        param_grid = {'max_depth': [None, 5, 10, 15]}  # 决策树参数网格
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')  # 网格搜索
        grid_search.fit(X_train, y_train)  # 训练模型
        best_model = grid_search.best_estimator_  # 获取最佳模型
    else:  # SVM
        param_grid = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1]}  # SVM参数网格
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')  # 网格搜索
        grid_search.fit(X_train, y_train)  # 训练模型
        best_model = grid_search.best_estimator_  # 获取最佳模型

    best_models[name] = best_model  # 保存最佳模型

    # 记录调优后的性能指标
    y_pred_post = best_model.predict(X_test)  # 预测测试集
    mse_post = mean_squared_error(y_test, y_pred_post)  # 计算均方误差
    r2_post = r2_score(y_test, y_pred_post)  # 计算R2评分
    mae_post = mean_absolute_error(y_test, y_pred_post)  # 计算平均绝对误差
    post_optimization_scores[name] = {"MSE": mse_post, "R2": r2_post, "MAE": mae_post}

# 输出每个模型的调优后性能指标
for name, scores in post_optimization_scores.items():
    print(f"\nPost-Optimization Scores for {name}:")
    print(f"MSE: {scores['MSE']}, R2: {scores['R2']}, MAE: {scores['MAE']}")

# 可视化调优后的性能
plt.figure(figsize=(10, 6))
post_optimization_r2_scores = [scores['R2'] for scores in post_optimization_scores.values()]
plt.bar(list(best_models.keys()), post_optimization_r2_scores, color='blue')
plt.title("Post-Optimization Model Performance (R2 Score)")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.ylim(0, 1)
plt.show()

# 选出具有最高 R2 分数的模型
best_model_names = [name for name, scores in post_optimization_scores.items() if
                    scores['R2'] == max(post_optimization_r2_scores)]
print("Best Model(s):", best_model_names)