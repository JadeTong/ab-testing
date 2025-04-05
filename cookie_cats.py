'''
数据集来自一款叫做《cookie cats》的三消游戏，当玩家玩到一定关卡时，强制要求他们等一段时间或者在app内购买充值包才能继续玩下一关。
下面，将对第一个‘gate’设置在第30个关卡还是第40个关卡进行A/B Test；细致地来说，针对用户留存率来分析。
分析发现将gate定在30关和定在40关，对一日留存率并没有影响，对七天留存率有略微影响

'''
#%% 读取数据
import pandas as pd
df = pd.read_csv('D:/desktop/ds/ab test/cookie_cats.csv')
df.head()
#%% 筛选出到达至少level 30的玩家
data = df[df.sum_gamerounds >= 30].copy()
print('30关以上的玩家有：',data.shape[0])
# 30关以上的玩家有： 33269

#%% 计算gate30和gate40两组的样本量
data['version'].value_counts()
# ==========================================================
# version
# gate_30    16656
# gate_40    16613
# ==========================================================
# 两组的样本量大致相等，只相差43。
#%% 计算这些玩家的平均关卡
data.groupby('version')['sum_gamerounds'].describe()
# gate30这一组的最大值是四万多？异常值吧。两组的平均都在125左右
#%% 检查一下数据是否有缺失和异常值
#%%%
data.isnull().sum()  #每一列都是0，lucky
#%%% 画个箱型图看一下gameround的异常值
data.boxplot(column='sum_gamerounds', by='version')
# 箱型图也能看出gate30那组有一个异常值，使正常数值分布变得扁平
data = data[data['sum_gamerounds']<max(data['sum_gamerounds'])]
data.boxplot(column='sum_gamerounds', by='version')

#%% 现在一天留存和七天留存都是以字符串形式储存的，将他们都换成数值形，方便计算均值。
data["retention_1"] = data['retention_1'].astype(int) #True=1, False=0
data['retention_7'] = data['retention_7'].astype(int)

#%%% 计算一天和七天的留存率
retention_1 = data.groupby('version')['retention_1'].mean()
print(retention_1)
# =========================================================
# version
# gate_30    0.801009
# gate_40    0.801842
# =========================================================
effect_size_1 = retention_1['gate_40'] - retention_1['gate_30']
print(effect_size_1)   #0.000833164355213966,这么小，不太可能显著差异吧，不过是正向的喔
retention_7 = data.groupby('version')['retention_7'].mean()
print(retention_7)
# ============================================================
# version
# gate_30    0.438693
# gate_40    0.430025
# ==========================================================
effect_size_7 = retention_7['gate_40'] - retention_7['gate_30']
print(effect_size_7)    #-0.008668727521425279，负值，gate30效果还好一点，不过也还是小啊，0.87%
#不管是一天留存还是七天留存，控制组和处理组的均值都非常相近

#%% 将数据集分割成控制组和处理组
gate30 = data[data.version == 'gate_30']
gate40 = data[data.version == 'gate_40']
#%% 统计检验（t检验）
#用t检验是因为retention_1和retention_7是可以视为连续变量：0/1的二元变量，当样本量足够大时，根据中心极限定理，0/1的均值近似服从正态分布，可以用t检验比较两个均值是否有显著差异。

#%%% 在t检验之前，先检验一下两组样本的方差是否一致（levene检验）
from scipy.stats import levene

stat_1, p_1 = levene(gate30['retention_1'], gate40['retention_1'])
print("Levene 检验的 P 值:", p_1)
#Levene 检验的 P 值: 0.8489509765880116
stat_7, p_7 = levene(gate30['retention_7'], gate40['retention_7'])
print("Levene 检验的 P 值:", p_7)
#Levene 检验的 P 值: 0.11073879416302806
#两个指标的两组数据都符合方差齐性，student's t-test（方差不齐，用welch's t-test）

#%%%
from scipy import stats
t_stat_1, p_value_1 = stats.ttest_ind(
  gate30['retention_1'], gate40['retention_1'], 
  equal_var=True)
print(f"T-statistic: {t_stat_1}, P-value: {p_value_1}")
# T-statistic: -0.19045892880340554, P-value: 0.8489506863561129
# p值远大于0.05，只能接受原假设，gate设置在30关和设置在40关，对1天留存率没有显著影响。

t_stat_7, p_value_7 = stats.ttest_ind(
  gate30['retention_7'], gate40['retention_7'], 
  equal_var=True)
print(f"T-statistic: {t_stat_7}, P-value: {p_value_7}")
# T-statistic: 1.5949282747308886, P-value: 0.11073777593144853
# p值为0.11，也大于0.05，接受原假设，gate设置在30关和设置在40关，对7天留存率没有显著影响。 ！但是现实情况可能不会设α为0.05吧，这么严格吗？可不可以算gate设置在不同关卡对7天留存率some what有影响呢？

#对于七天留存率，用单尾检验试一下
t_stat_7, p_value_7 = stats.ttest_ind(
  gate30['retention_7'], gate40['retention_7'], 
  equal_var=False, alternative='greater')
print(f"T-statistic: {t_stat_7}, P-value: {p_value_7}")
# T-statistic: 1.5949282747308886, P-value: 0.05536888796572426

'''
单尾t检验假设 “gate_30 的留存率会高于 gate_40”。
p值略高于传统显著性阈值 0.05 ⇒ “不显著”，但很接近边界。认为我们有边缘性证据支持 gate_30 的 7 天留存率更好，但不足以下结论。(如果将置信水平改为0.1，就可以认为有显著差异)
anyway，游戏公司应该将gate把持在关卡30，不需要更改到关卡40，反正改到gate40肯定没有gate30的留存率高。
'''

#%% 算一下Cohen’s d值，
# 计算合并标准差 pooled standard deviation
group_30 = gate30["retention_7"]
group_40 = gate40["retention_7"]
import numpy as np
n1, n2 = len(group_30), len(group_40)
s1, s2 = group_30.std(), group_40.std()
pooled_std = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))

# 计算 cohen's d
cohen_d = (group_30.mean() - group_40.mean()) / pooled_std
print("Cohen's d =", cohen_d)   #Cohen's d = 0.017624765545438226

'''
Cohen's d = 0.0176, 这个值非常小，远低于 0.2（小效应的下限）。
说明即使差异显著，它的实际影响力也很小，业务意义有限。

虽然在7天留存率上，gate_30相比gate_40的t检验结果接近显著（p = 0.055，单尾），但效果量（Cohen’s d = 0.0176）极小，说明即便差异存在，其实际业务影响也非常有限。因此，目前没有充分理由建议基于此指标切换版本。
'''

#%%
'''
其实我觉得这个数据集对练习项目来说不是很好，数据信息不多，可做的步骤没几个，比较简单，继续找其它数据集来练练先
'''






