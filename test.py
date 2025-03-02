import numpy as np
import modin.pandas as pd 
class MyNN:
    def __init__(self, input_size, layer_number, layer_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.layer_number = layer_number
        self.layer_size = layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.matrices_list = []
        self.matrices_list.append(np.random.randn(input_size, layer_size) * 0.01)
        
        for _ in range(layer_number - 2):
            self.matrices_list.append(np.random.randn(layer_size, layer_size) * 0.01)
        
        self.matrices_list.append(np.random.randn(layer_size, output_size) * 0.01)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, input):
        layer_output = [input]
        for i, matrice in enumerate(self.matrices_list):
            output = layer_output[-1] @ matrice
            if i == len(self.matrices_list) - 1:
                output = self.softmax(output)  
            else:
                output = self.relu(output) 
            layer_output.append(output)
        return output, layer_output

    def compute_error(self, prediction, target):
        error = prediction - target
        mse = np.mean(error ** 2) 
        return mse, error

    def backward(self, layer_output, input, error):
        grad_matrices = []
        delta = error * self.relu_derivative(layer_output[-1])

        for i in reversed(range(len(self.matrices_list))):
            grad = layer_output[i].T @ delta
            grad_matrices.append(grad)
            delta = (delta @ self.matrices_list[i].T) * self.relu_derivative(layer_output[i])
        grad_matrices.reverse()
        return grad_matrices

    def update_weights(self, grad_matrices):
        for i in range(len(self.matrices_list)):
            self.matrices_list[i] -= self.learning_rate * grad_matrices[i]

    def train(self, x_train, y_train, epochs=1000):
        for epoch in range(epochs):
            output, layer_output = self.forward(x_train)
            mse, error = self.compute_error(output, y_train)
            grads = self.backward(layer_output, x_train, error)
            self.update_weights(grads)
            print(f"Epoch {epoch}, Loss: {mse}")

def load_csv(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        data = np.array([[float(num) for num in line.strip().split(",")] for line in lines])
    np.random.shuffle(data)
    x_data = [row[:784] for row in data]
    y_data = [row[784:] for row in data]
    return x_data, y_data

csv_path = "images_with_extra.csv"  # Change to actual CSV file path
x_train, y_train = load_csv(csv_path)

x_train = np.array(x_train)
y_train = np.array(y_train)

nn = MyNN(input_size=784, layer_number=3, layer_size=128, output_size=10,learning_rate=0.01)
nn.train(x_train, y_train, epochs=200)
