"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H_1,..., H_{k-1}] with the number of neurons H_i in the hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        

            # TODO: (Extra Credit) You may set parameters for Adam optimizer here
        # TODO: (Extra Credit) You may set parameters for Adam optimizer here
        if opt == 'Adam':
            self.t = 0  # timestep
            self.m = {}  # first moment
            self.v = {}  # second moment
            for i in range(1, num_layers + 1):
                self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
                self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
                self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        Z = X @ W + b
        return Z
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        # TODO: implement me
        de_dw = X.T @ de_dz
        de_db = np.sum(de_dz, axis=0)
        de_dx = de_dz @ W.T
        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return (X > 0).astype(float)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
         
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        # Here, X can be either the input to the sigmoid or the output of the
        # sigmoid, depending on your implementation.
        s = self.sigmoid(X)
        grd = s * (1 - s)
        return grd

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this. Your MSE loss should be implemented as:
        # MSE = $0.5 \frac{1}{NC} \sum \limits_{i=1}^N \sum \limits_{j=1}^C (y_{ij} - p_{ij})^2$
        # where y and p are of shape (N, C).
        return 0.5 * np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        N , C = y.shape
        return (p - y) / (N * C)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """

        self.inputs = X # For use later during backpropagation
        self.outputs = {}

        # TODO: implement me. You'll want to store the output of each layer in 
        # self.outputs as they will be used during backpropagation. You can use 
        # functions like self.linear and self.relu to help organize your code.
        current_input = X
        for i in range(1, self.num_layers):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            z = self.linear(W, current_input, b)
            self.outputs["Z" + str(i)] = z
            a = self.relu(z)
            self.outputs["A" + str(i)] = a
            current_input = a
        # Last layer
        W = self.params["W" + str(self.num_layers)]
        b = self.params["b" + str(self.num_layers)]
        z = self.linear(W, current_input, b)
        self.outputs["Z" + str(self.num_layers)] = z
        a = self.sigmoid(z)
        self.outputs["A" + str(self.num_layers)] = a

        return a

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss
        """

        

        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.sigmoid_grad if it helps organize your code.
        self.gradients = {}
        p = self.outputs["A" + str(self.num_layers)]
        loss = self.mse(y, p)
        de_dp = self.mse_grad(y, p)
        de_dz = de_dp * self.sigmoid_grad(self.outputs["Z" + str(self.num_layers)])

        for i in range(self.num_layers, 0, -1):
            W = self.params["W"+ str(i)]
            X = self.inputs if i == 1 else self.outputs["A" + str(i - 1)]
            de_dw, de_db, de_dx = self.linear_grad(W, X, de_dz)
            self.gradients["W" + str(i)] = de_dw
            self.gradients["b" + str(i)] = de_db

            if i > 1:
                de_dz = de_dx * self.relu_grad(self.outputs["Z" + str(i - 1)])

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for i in range(1, self.num_layers+1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]


            pass
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            self.t += 1  # increment timestep
            for i in range(1, self.num_layers + 1):
                for key in ["W" + str(i), "b" + str(i)]:
                    g = self.gradients[key]
                    
                    # Actualize moments
                    self.m[key] = b1 * self.m[key] + (1 - b1) * g
                    self.v[key] = b2 * self.v[key] + (1 - b2) * g ** 2
                    
                    # bias correction
                    m_hat = self.m[key] / (1 - b1 ** self.t)
                    v_hat = self.v[key] / (1 - b2 ** self.t)
                    
                    # Update final
                    self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            
            pass
        else:
            raise NotImplementedError
        