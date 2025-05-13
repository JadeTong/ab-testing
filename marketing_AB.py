'''
Digital Marketing A/B Testing
一家marketing公司对他们的广告营销结果做ab测试，用来测评广告营销的效果。
ab测试比较看到广告的用户和看到psa的用户之间的转化率（conversion rate）

可以尝试解决的问题：
* 多少的转化率是归因于营销的呢？
* ad组和psa组的转化率有统计学上的显著差异吗？
* 效果量effect size是多少（ad组比控制组效果好/差多少）？
* 若两组转化率存在显著差异，差异的置信区间为多少？

* 观看广告的数量与转化率之间有相关关系吗？
* 提升转化率的最佳广告观看数量是多少？

* 一周中的星期几有最高的转化率？
* 不同日期的转化率之间有显著差异吗？（F检验？）


应该要得出得结果：
1. 验证营销策略在统计学上的显著性
2. 用置信区间来量化business impact
3. 推荐最佳广告推送时间
4. 往后营销策略的优化
'''

#%% 读取数据
import pandas as pd
df = pd.read_csv('D:/desktop/ds/ab test/marketing_AB.csv',index_col=0)
df.head()

#%% 检查一下缺失值，异常值
print(df.isnull().sum()) #没有缺失值
#看一下广告推送量有没有异常值
quantile = df.groupby('test group')['total ads'].describe()
print(quantile)
# =============================================================================
#                count       mean        std  min  25%   50%   75%     max
# test group                                                              
# ad          564577.0  24.823365  43.750456  1.0  4.0  13.0  27.0  2065.0
# psa          23524.0  24.761138  42.860720  1.0  4.0  12.0  26.0   907.0
# =============================================================================
#被推送广告的用户有564577位，被推送psa的用户有23524位，两组用户被推送的平均次数都是大约为24次，广告组最多次数为2065，psa组最多次数为907，画一下直方图看下分布，再考虑要不要剔除异常值。
ads = df[df['test group'] == 'ad']
psa = df[df['test group'] == 'psa']

import matplotlib.pyplot as plt
_ = plt.hist(ads['total ads'], bins=100, alpha=0.5, label='Ads', edgecolor='black')    #_是为了不输出返回值array
_ = plt.hist(psa['total ads'], bins=100, alpha=0.5, label='PSA', edgecolor='black')
#两组数据都极度右偏，特别长尾，考虑将top 1%剔除掉
limit_ads = ads['total ads'].quantile(0.99)
print('ads组的异常值数阈值:',limit_ads)  #201
outliers_ads = ads[ads['total ads'] >= limit_ads]
print('ads组的异常值数:',len(outliers_ads))  #5694

limit_psa = psa['total ads'].quantile(0.99)
print('psa组的异常值数阈值:',limit_psa)  #206
outliers_psa = psa[psa['total ads'] >= limit_psa]
print('psa组的异常值数:',len(outliers_psa))   #239

#%%% 重新定义ads组和psa组
ads = ads[ads['total ads'] < limit_ads]
psa = psa[psa['total ads'] < limit_psa]

_ = plt.hist(ads['total ads'], bins=100, alpha=0.5, label='Ads', edgecolor='black')
_ = plt.hist(psa['total ads'], bins=100, alpha=0.5, label='PSA', edgecolor='black')

#%%计算转化率差异，可视化
data = pd.concat([ads, psa], axis=0)  #axis=0,按行拼接
data['converted'] = data['converted'].astype(int)
data.groupby('test group')['converted'].mean()
data.groupby('test group')['converted'].value_counts()
conversion = data.groupby('test group')['converted'].agg(
        count='count',
        sum='sum',
        conversion_rate='mean'
        )
print(conversion)
# =============================================================
#              count    sum  conversion_rate
# test group                                
# ad          558883  13536         0.024220
# psa          23285    385         0.016534
# =============================================================
p1 = ads['converted'].mean()
p2 = psa['converted'].mean()
diff = p1-p2
print(f'\nads组和psa组的转化率差：{diff:.2%}')    #ads组和psa组的转化率差：0.77%
# 计算相对提升度
lift = diff/p2
print(f'相对提升度为：{lift:.2%}')    #相对提升度为：46.48%，还挺高啊

#%% 检验两组处理的转化率是否有显著差异，z检验或卡方检验
'''
原假设H0：ad组的转化率与psa组转化率相等   P(ad)-P(psa)=0
备择假设H1：ad组的转化率高于psa组的转化率（单尾） P(ad)-P(psa) > 0
'''
#%%% 用两比例Z检验（2 proportion Z-Test），先检验一下数据是否符合检验条件
# 1.样本独立性，同一用户不能出现在两个组中
#检查每个用户是否只出现在一个组中
duplicates = data.groupby('user id')['test group'].nunique()
non_independent_users = duplicates[duplicates > 1]
print(f"发现{len(non_independent_users)}个用户出现在多个组中")  #0个

#检查整个数据集中是否有重复用户
duplicate_users = data['user id'].duplicated().sum()
print(f"有{duplicate_users}个重复user id")  #0个

# 2.样本容量足够大，中心极限定理
#如果你的样本量够大，不管原始数据怎么分布，抽样多次的样本比例p̂的分布就会近似一个正态分布
# 检查样本量的标准是： n1p1>=5, n1(1-p1)>=5, n2p2>=5, n2(1-p2)>=5; 其实就是转化和未转化的用户都大于等于5,这两组数据都远大于5
n1 = len(ads)
p1 = ads['converted'].mean()
x1 = data.groupby('test group')['converted'].value_counts().loc[('ad', 1)]
ads_fail = data.groupby('test group')['converted'].value_counts().loc[('ad', 0)]
n2 = len(psa)
p2 = psa['converted'].mean()
x2 = data.groupby('test group')['converted'].value_counts().loc[('psa', 1)]
psa_fail = data.groupby('test group')['converted'].value_counts().loc[('psa', 0)]
print(data.groupby('test group')['converted'].value_counts())
# 3.二元响应变量 0/1变量，已满足

#%%% 两比例Z检验
from statsmodels.stats.proportion import proportions_ztest
success = [x1, x2]
nobs = [n1,n2]
z_stat, p_val = proportions_ztest(success, nobs, alternative='larger')
print(f"Z统计量: {z_stat:.4f}, p值: {p_val:.4f}")
#Z统计量: 7.5213, p值: 0.0000小于α=0.05，即在显著性水平为0.05下，拒绝原假设，认为ad组的转化率显著高于psa组。

#%% 转化率的95%置信区间 （即便转化率有差异，这个差值的可信范围是多少？）
# CI = (两组转化率之差 - z*标准差, 两组转化率之差 + z*标准差)，这里z选1.645，因为是单尾
import numpy as np
from scipy.stats import norm
se = np.sqrt((p1*(1-p1)/n1)+(p2*(1-p2)/n2))
z = norm.ppf(0.95)
ci_lower = (p1-p2) - z * se
ci_upper = (p1-p2) + z * se
print(f"转化率差值: {diff:.4f}")
print("95% 置信区间: [{:.4f}, {:.4f}]".format(ci_lower, ci_upper))
# 转化率差值: 0.0077
# 95% 置信区间: [0.0063, 0.0091]，说明ad组的转化率相对于有正向提升


#%% 计算效果量effect size，衡量差异是否有实际意义
# cohen's h = 2*(arcsin(p1的平方根) - arcsin(p2的平方根))
from math import asin, sqrt
cohens_h = 2*(asin(sqrt(p1) - asin(sqrt(p2))))
print("Cohen's h值: ",cohens_h)     #Cohen's h值:  0.05337501649888809
# h值约等于0.05 → 差异非常小（远低于0.2），说明这个差异在业务上可能并不重要，用户行为并没有实质改变，提升幅度极其有限。这个广告设计确实带来了变化，但用户并没多喜欢到哪儿去。     is there a point to keep doing the rest of the test?


#%% 对广告曝光量分析，（1）看广告量与转化是否有关联 （2）统计分析和可视化
#%%% 画个boxplot，看下转化与否相对应的广告量分布
import seaborn as sns
plt.figure(figsize=(8,6))

sns.boxplot(data=ads, x='converted', y='total ads')
#converted的用户所观看的广告量的中位数明显高于未转化用户，而且它的四分位距也比未转化的宽，说用户所观看的广告量会影响转化率。
ads.groupby('converted')['total ads'].describe()

#%%% 做个点二列相关分析 （一个连续变量，一个二分类变量）
from scipy.stats import pointbiserialr
r, p_val = pointbiserialr(ads['converted'], ads['total ads'])

print('相关系数：', r)   #约等于0.26
print('p值：', p_val)    #小于0.01
# 显著的弱正相关性

#%%% 找出最佳广告观看量，只对处理组ads组操作
#（1）对total ads分箱，比较每一段的转化率
ads['total ads'].describe()
#等宽分箱
ads['bin'] = pd.cut(ads['total ads'], bins=range(0, 201, 10), right=True)
conversion_by_bin = ads.groupby('bin')['converted'].agg(
    count='count', converted='sum', conversion_rate='mean'
).reset_index()

plt.figure(figsize=(12,6))
plt.plot(conversion_by_bin['bin'].astype(str), conversion_by_bin['conversion_rate'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('ads interval')
plt.ylabel('conversion rate')
plt.title('num of ads vs conversion rate')
plt.grid(True)



#等频分箱
ads['bin'] = pd.qcut(ads['total ads'], q=10, duplicates='drop')
conversion_by_bin = ads.groupby('bin')['converted'].agg(
    count='count', converted='sum', conversion_rate='mean'
).reset_index()

plt.figure(figsize=(12,6))
plt.plot(conversion_by_bin['bin'].astype(str), conversion_by_bin['conversion_rate'], marker='o')
plt.xticks(rotation=45)
plt.xlabel('ads interval')
plt.ylabel('conversion rate')
plt.title('num of ads vs conversion rate')
plt.grid(True)
#最后一箱 (53.0, 200.0] 虽然转化率很高，但这部分人已经是 “极端高曝光”用户，可能有其他因素（如对产品本身感兴趣、定向广告投放更精准）在影响转化。
max_row = conversion_by_bin.loc[conversion_by_bin['conversion_rate'].idxmax()]
print("转化率最高的广告区间：", max_row['bin'])
print("该区间转化率：{:.2%}".format(max_row['conversion_rate']))

#%%% 逻辑回归，加入二次项来捕捉非线性关系，也可以先拟合一下加入二次项和不加二次项的回归模型来检验是否有必要
#一元一次
import statsmodels.api as sm

X = ads[['total ads']]
X = sm.add_constant(X)
y = ads['converted']
model = sm.Logit(y, X).fit()
print(model.summary())

#加入平方项
X = ads[['total ads']]
X['ads_squared'] = X['total ads'] ** 2
X = sm.add_constant(X)
y = ads['converted']
model = sm.Logit(y, X).fit()
print(model.summary())
#模型伪R方约等于0.2，拟合效果好，total_ads和ads_squared的p值都小于0.001，平方项的系数值很小，但是乘上 ads² 后仍可能有实质性的非线性影响。


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = ads[['total ads']]
X['ads_squared'] = X['total ads'] ** 2
y = ads['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)




ads_range = np.arange(1, 201)
ads_squared = ads_range ** 2

# 组装为模型需要的输入形式
X_pred = sm.add_constant(pd.DataFrame({
    'total ads': ads_range,
    'ads_squared': ads_squared
}))

# 预测概率
pred_probs = model.predict(X_pred)

# 找到最大概率的位置
best_ads = ads_range[np.argmax(pred_probs)]
best_prob = np.max(pred_probs)

print(f"转化率最高出现在广告数 = {best_ads}，其预测转化率为：{best_prob:.4f}")


























