import torch
import torch.nn as nn
import torch.optim as optim


# 定义多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 输入层到隐藏层
        self.fc2 = nn.Linear(4, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.fc1(x)  # 第一层
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 第二层
        return torch.sigmoid(x)  # Sigmoid 激活，输出 0 或 1


# 准备训练数据
data = torch.tensor([[35, 67], [12, 75], [16, 89], [45, 56], [10, 90]], dtype=torch.float32)  # 学习时间和期中成绩
labels = torch.tensor([[1], [0], [1], [1], [0]], dtype=torch.float32)  # 期末考试通过情况

# 初始化模型
model = MLP()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # 清空上次的梯度
    outputs = model(data)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 预测学习时间为 25 小时，期中成绩为 70 的学生
test_data = torch.tensor([[25, 70]], dtype=torch.float32)  # 测试数据
model.eval()  # 设置为评估模式
with torch.no_grad():  # 关闭梯度计算
    prediction = model(test_data)
    predicted_class = prediction.round()  # 将输出四舍五入为 0 或 1

print(f"预测结果: {'通过' if predicted_class.item() == 1 else '失败'}")
