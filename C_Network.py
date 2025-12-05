import numpy as np
from typing import List, Callable, Tuple


# ---- Fonctions d'activation ----
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoïde appliquée élément-par-élément."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative_from_activation(a: np.ndarray) -> np.ndarray:
    """
    Dérivée de la sigmoïde en fonction de l'activation a = sigmoid(z).
    d(sigmoid)/dz = a * (1 - a)
    """
    return a * (1.0 - a)


# ---- Classe Layer (couche dense) ----
class Layer:
    """
    Couche fully-connected (dense).
    - weights: shape (output_size, input_size)
    - biases:  shape (output_size,)
    - last_output: activation de la couche après forward, shape (output_size,)
    """
    def __init__(self, input_size: int, output_size: int,
                 activation: Callable[[np.ndarray], np.ndarray] = sigmoid,
                 activation_derivative: Callable[[np.ndarray], np.ndarray] = sigmoid_derivative_from_activation):
        self.input_size = input_size
        self.output_size = output_size
        # Initialisation des poids (Xavier / Glorot pour sigmoïde)
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, size=(output_size, input_size))
        self.biases = np.zeros(output_size)
        # stocke l'activation de la couche après forward (utile pour backprop)
        self.last_output: np.ndarray = np.zeros(output_size)

        # fonctions d'activation
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagation avant d'une couche:
        z = W @ x + b  (W shape (out, in), x shape (in,))
        a = activation(z)
        """
        z = self.weights @ x + self.biases  # produit matriciel + biais
        a = self.activation(z)
        self.last_output = a
        return a


# ---- Classe NeuralNetwork ----
class NeuralNetwork:
    """
    Réseau constitué d'une liste de Layer.
    Exemple d'initialisation : NeuralNetwork([Layer(2,4), Layer(4,1)])
    """
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.activations: List[np.ndarray] = []  # stocke activations intermédiaires après forward

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward complet : renvoie la sortie du réseau et stocke toutes les activations.
        activations[0] = entrée (x)
        activations[1] = sortie couche 0
        ...
        activations[-1] = sortie finale
        """
        self.activations = [x.copy()]
        out = x
        for layer in self.layers:
            out = layer.forward(out)
            self.activations.append(out.copy())
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Applique forward à chaque exemple (X shape (n_samples, n_features))."""
        outs = []
        for x in X:
            outs.append(self.forward(x))
        return np.vstack(outs)

    def train(self, X: np.ndarray, y: np.ndarray,
              lr: float = 0.1, epochs: int = 1000, verbose: bool = False) -> None:
        """
        Entraînement par SGD (itère chaque échantillon, mise à jour immédiate).
        - X : shape (n_samples, input_size)
        - y : shape (n_samples, output_size)
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            for i in range(n_samples):
                x = X[i]
                target = y[i]

                # ---- Forward ----
                output = self.forward(x)  # shape (output_size,)

                # ---- Calcul du delta initial de la couche de sortie ----
                # Si on utilise la perte MSE L = 0.5 * ||output - target||^2,
                # dL/doutput = output - target.
                delta = output - target  # shape (output_size,)

                # ---- Backprop couche par couche (de la sortie vers l'entrée) ----
                for l in reversed(range(len(self.layers))):
                    layer = self.layers[l]
                    a_prev = self.activations[l]  # activation de la couche précédente (a^{l-1}), shape (input_size,)

                    # Gradient des poids: grad_W = delta[:, None] * a_prev[None, :]
                    # Ce qui donne une matrice de shape (output_size, input_size).
                    grad_w = np.outer(delta, a_prev)  # shape (out, in)
                    grad_b = delta.copy()              # shape (out,)

                    # Mise à jour (descente de gradient)
                    layer.weights -= lr * grad_w
                    layer.biases -= lr * grad_b

                    # Si couche précédente existe, calculer delta précédent
                    if l > 0:
                        # propagation de l'erreur: delta_prev = (W^T @ delta) * activation'(a_prev)
                        delta = (layer.weights.T @ delta) * layer.activation_derivative(a_prev)

            if verbose and (epoch % (epochs // 10 + 1) == 0):
                # calcul simple de la perte MSE sur tout le dataset pour monitoring
                preds = self.predict(X)
                loss = np.mean(0.5 * (preds - y) ** 2)
                print(f"epoch {epoch}/{epochs} - loss {loss:.6f}")

# ====================================
#          TEST DU XOR (ICI)
# ====================================


if __name__ == "__main__":
    nn = NeuralNetwork([
    Layer(2, 4),  # 4 neurones au lieu de 2
    Layer(4, 1)
])

    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]], dtype=float)

    y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=float)

    nn.train(X, y, lr=0.5, epochs=10000)

    print("\n=== TESTS FINALS ===")
    for x in X:
        print(f"{x} → {nn.forward(x)}")