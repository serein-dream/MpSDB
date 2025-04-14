import matplotlib.pyplot as plt
import numpy as np

# 数据集规模
dataset_sizes = [3 * 3, 10 * 3, 50 * 3]
type_now = 'mean'
# mean的平均时长数据
if type_now == 'mean':
    mean_duration_system_A = [124.0234, 398.045, 3270.404]
    mean_duration_system_B = [0.06725,	0.07075,	0.071]
# std_dev的平均时长数据
else:
    mean_duration_system_A = [0.64044,	0.56022,	0.590816]
    mean_duration_system_B = [86.67266667,	1686.141667, 0]

# 创建图表
plt.figure(figsize=(12, 8), dpi=1000)

# 绘制折线图
plt.plot(dataset_sizes, mean_duration_system_A, marker='o', label='Conclave', color='blue')
plt.plot(dataset_sizes, mean_duration_system_B, marker='o', label='Ours', color='red')

# 添加数据标签
for i, txt in enumerate(mean_duration_system_A):
    plt.annotate(f'{txt:.4f}', (dataset_sizes[i], mean_duration_system_A[i] + 0.2), ha='center', va='bottom', color='blue')

for i, txt in enumerate(mean_duration_system_B):
    plt.annotate(f'{txt:.4f}', (dataset_sizes[i], mean_duration_system_B[i] + 0.2), ha='center', va='bottom', color='red')

# 设置图表标题和标签
plt.title('Comparison of' + type_now)
plt.xlabel('Dataset Size')
plt.ylabel('Run time(seconds)')

# 添加图例
plt.legend()

# 显示网格
plt.grid(False)
# 设置横坐标刻度
plt.xticks(dataset_sizes, ['3x3', '10x3', '50x3'])
plt.savefig(type_now+'.png')
