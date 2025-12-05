import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases  = np.zeros((output_size,))
    
    def forward(self, x):
        self.last_output = sigmoid(self.weights @ x + self.biases)
        return self.last_output

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def train(self, X, y, lr=0.5, epochs=10000):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                # Forward
                activations = [xi]
                out = xi
                for layer in self.layers:
                    out = layer.forward(out)
                    activations.append(out)
                
                # Backprop
                delta = out - yi
                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    a_prev = activations[i]
                    grad_w = np.outer(delta, a_prev)
                    grad_b = delta
                    layer.weights -= lr * grad_w
                    layer.biases  -= lr * grad_b
                    if i > 0:
                        delta = (layer.weights.T @ delta) * sigmoid_derivative(a_prev)

# Exemple d'utilisation (XOR)
if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    nn = NeuralNetwork([Layer(2,4), Layer(4,1)])
    nn.train(X, y)
    for x in X:
        print(x, "→", nn.forward(x))
