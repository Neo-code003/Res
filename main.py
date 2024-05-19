import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

data = pd.read_csv('covid.train.csv')
all_positive = data.filter(like='tested_positive')


# 共5个 最后一个是预测值

# x_train_1 = all_positive['tested_positive.1']
# x_train_2 = all_positive['tested_positive.2']
# y_train = all_positive['tested_positive.3']
#
# print(x_train_1)
# print(x_train_2)
# print(y_train)
# print(all_positive)

# class Model(nn.Module):
#     def __init__(self, input_dim):
#         super(Model, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1)
#         )
#
#     def forward(self, x):
#         x = self.layers(x)
#         x = x.squeeze(1)
#         return x

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

x_train = all_positive.drop(columns=['tested_positive.4']).values
y_train = all_positive['tested_positive.4'].values

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
# 转置 变成竖列 [,,,]->[[]]

# print(x_train)
# print(y_train)

# print(x_train.shape[1])

model = Model(input_dim=x_train.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
grads = []

num_epochs = 1000
batch_size = 32
dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        # print(outputs)
        # print(batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    model.train()
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.norm().item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失值和梯度值折线图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(False)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(grads) + 1), grads, label='Gradients')
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm over Training Steps')
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()


test_data = pd.read_csv('covid.test.csv')
x_test = test_data.filter(like='tested_positive').values
x_test = torch.tensor(x_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred = model(x_test).squeeze().numpy()

df = pd.DataFrame({'id': test_data['id'], 'tested_positive': y_pred})
df.to_csv('predicted_tested_positive.csv', index=False)
