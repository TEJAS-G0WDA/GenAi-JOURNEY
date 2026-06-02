import torch
import torch.nn as nn
import torch.optim as optim

# Dataset
X = torch.tensor([
    [1.0, 2.0],
    [2.0, 1.0],
    [5.0, 4.0],
    [6.0, 5.0]
])

y = torch.tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])

# Neural Network
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
    nn.Sigmoid()
)

# Loss Function
loss_function = nn.BCELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# Training
for epoch in range(100):

    predictions = model(X)

    loss = loss_function(
        predictions,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}, Loss = {loss.item():.4f}"
    )