
import ast
import json
import matplotlib.pyplot as plt

losses = []
with open('1.txt','r') as f:
    for i in f.readlines() :
        if i[0] == '{' :
            dict = json.loads(i)
            losses.append(dict['loss'])


# 打印所有loss值
print("Loss values:")
for loss in losses:
    print(loss)

# 绘制loss曲线图
steps = range(1, len(losses) + 1)  # 假设步骤是从1开始的连续整数
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o', linestyle='-', color='b')
plt.title('Loss During Training')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()