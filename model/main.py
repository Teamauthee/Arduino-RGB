import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from nn import ColorClassifier

# Data Prep
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "data", "colors_labeled.csv")
df = pd.read_csv(csv_path)

x = df[["red", "green", "blue"]].values
x_scaled = x / 255.0

y = df["label"].values

X = torch.from_numpy(x_scaled).float()
Y = torch.from_numpy(y).long()

input_dim = X.shape[1]
output_dim = 3

model = ColorClassifier(input_dim, output_dim)

# Training
epochs = 200
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()

    outputs = model(X.float())
    loss = criterion(outputs, Y.view(-1).long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == Y.view(-1)).sum().item()
        accuracy = 100 * correct / Y.size(0)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

# Inference for Arduino (C++)
model.eval()

def fold_batchnorm(fc, bn):
    w = fc.weight.detach().numpy()
    b = fc.bias.detach().numpy()

    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.numpy()
    var = bn.running_var.numpy()
    eps = bn.eps

    std = np.sqrt(var + eps)
    multiplier = gamma / std

    w_folded = w * multiplier[:, None]
    b_folded = (b - mean) * multiplier + beta
    return w_folded, b_folded

w1, b1 = fold_batchnorm(model.fc1, model.bn1)
w2, b2 = fold_batchnorm(model.fc2, model.bn2)

w3 = model.fc3.weight.detach().numpy()
b3 = model.fc3.bias.detach().numpy()

def to_cpp_array(name, array):
    flat = array.flatten()
    values = ", ".join([f"{val:.6f}" for val in flat])
    return f"const float {name}[{len(flat)}] PROGMEM = {{{values}}};\n"

cpp_code = "#include <avr/pgmspace.h>\n\n"
cpp_code += to_cpp_array("W1", w1)
cpp_code += to_cpp_array("b1", b1)
cpp_code += to_cpp_array("W2", w2)
cpp_code += to_cpp_array("b2", b2)
cpp_code += to_cpp_array("W3", w3)
cpp_code += to_cpp_array("b3", b3)

print(cpp_code)