import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    """
    A reusable library for Gradient Descent variants.
    """
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, 
                 tolerance: float = 1e-6, method: str = 'batch', batch_size: int = 32):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance
        self.method = method # 'batch', 'stochastic', 'mini_batch'
        self.batch_size = batch_size
        self.theta = None
        self.cost_history = []
        
    def _compute_cost(self, X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        return (1/(2*m)) * np.sum(np.square(predictions - y))

    def _gradient(self, X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        return (1/m) * X.T.dot(predictions - y)

    def fit(self, X, y):
        """
        Fits the model to the data. 
        X: Feature matrix (m x n)
        y: Target vector (m,)
        """
        m, n = X.shape
        self.theta = np.zeros(n) # Initialize theta
        self.cost_history = []
        
        for i in range(self.iterations):
            if self.method == 'batch':
                grad = self._gradient(X, y, self.theta)
                self.theta -= self.learning_rate * grad
                
            elif self.method == 'stochastic':
                idx = np.random.randint(m)
                X_i = X[idx:idx+1]
                y_i = y[idx:idx+1]
                # SGD gradient (no 1/m)
                prediction = X_i.dot(self.theta)
                grad = X_i.T.dot(prediction - y_i)
                self.theta -= self.learning_rate * grad
                
            elif self.method == 'mini_batch':
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                # Update for one batch only per iteration (or full epoch? usually loop batches)
                # Here we do full epoch loop for mini_batch inside the iter check?
                # To keep it consistent with "iterations" usually meaning "updates" in SGD, 
                # but "epochs" in Batch. 
                # Let's assume 'iterations' here means 'Epochs' for simplified consistency.
                
                for start in range(0, m, self.batch_size):
                    end = start + self.batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    
                    # Mini-batch gradient
                    # Re-use _gradient but with subset
                    grad = self._gradient(X_batch, y_batch, self.theta)
                    self.theta -= self.learning_rate * grad
            
            # Record cost
            cost = self._compute_cost(X, y, self.theta)
            self.cost_history.append(cost)
            
            # Simple convergence check (for batch mostly)
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged at iteration {i}")
                break
                
        return self.theta

    def predict(self, X):
        return X.dot(self.theta)

if __name__ == "__main__":
    # Test the library
    print("Testing GradientDescentLibrary...")
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # Add bias
    X_b = np.c_[np.ones((100, 1)), X]
    
    gd = GradientDescent(learning_rate=0.1, iterations=100, method='batch')
    theta = gd.fit(X_b, y.flatten())
    
    print(f"Fitted Theta (Batch): {theta}")
    
    plt.plot(gd.cost_history)
    plt.title("Cost History (Batch)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show() # Blocks
