import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def load_csv(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        data = np.array([[float(num) for num in line.strip().split(",")] for line in lines])

    np.random.shuffle(data)

    x_data = torch.tensor(data[:, :784], dtype=torch.float32)  # Convert to tensor
    y_data = torch.tensor(data[:, 784:], dtype=torch.float32)  # Convert to tensor

    return x_data, y_data


csv_path = "images_with_extra.csv"  
x_train, y_train = load_csv(csv_path)

model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Sigmoid()
)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, x_train, y_train, criterion, optimizer, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()  
        output = model(x_train)  
        loss = criterion(output, y_train) 
        loss.backward()  
        optimizer.step()  
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Run training
train(model, x_train, y_train, criterion, optimizer)

torch.save(model, 'model_v2.pth')