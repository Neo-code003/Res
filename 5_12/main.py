import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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


train_data = pd.read_csv('covid.train.csv')
X_train = train_data.drop(columns=['tested_positive']).values
y_train = train_data['tested_positive'].values

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

model = Model(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []

num_epochs = 1000
batch_size = 32
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

test_data = pd.read_csv('covid.test.csv')
X_test = test_data.values

X_test = torch.tensor(X_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()

df = pd.DataFrame({'id': test_data['id'], 'tested_positive': y_pred})
df.to_csv('predicted_tested_positive.csv', index=False)
