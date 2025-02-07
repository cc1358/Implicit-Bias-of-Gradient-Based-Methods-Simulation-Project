"""
Implicit Bias of Gradient-Based Methods: Simulation Project
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import svd, orth
from tqdm import tqdm

class ImplicitBiasAnalyzer:
    def __init__(self, n=50, p=100, noise_std=0.1, loss_type='least_squares'):
        """
        Initialize underdetermined linear system with:
        - n samples, p features (p > n)
        - Ground truth model with sparse coefficients
        - Multiple global minima through underdetermined system
        """
        self.n, self.p = n, p
        self.A = np.random.randn(n, p)  # Data matrix
        self.A = self.A / np.linalg.norm(self.A, axis=1, keepdims=True)  # Normalize rows
        self.x_true = np.random.randn(p) * (np.random.rand(p) > 0.9)  # Sparse true coefficients
        self.b = self.A @ self.x_true + noise_std * np.random.randn(n)
        
        # Compute critical subspaces
        self.U, self.s, self.Vh = svd(self.A, full_matrices=False)
        self.P = self.Vh.T @ self.Vh  # Projection onto row space
        self.P_orth = np.eye(p) - self.P  # Orthogonal complement
        
        # Loss function configuration
        self.loss_type = loss_type
        self.losses = {
            'least_squares': lambda z, y: 0.5*(z-y)**2,
            'huber': lambda z, y: np.where(np.abs(z-y) < 1, 0.5*(z-y)**2, np.abs(z-y)-0.5)
        }
    
    def grad(self, x, batch_size=None):
        """Compute gradient with optional minibatch"""
        if batch_size is None:
            idx = slice(None)
        else:
            idx = np.random.choice(self.n, batch_size, replace=False)
            
        residuals = self.A[idx] @ x - self.b[idx]
        
        if self.loss_type == 'least_squares':
            return self.A[idx].T @ residuals
        elif self.loss_type == 'huber':
            mask = np.abs(residuals) < 1
            grad_coeffs = np.where(mask, residuals, np.sign(residuals))
            return self.A[idx].T @ grad_coeffs
    
    def optimize(self, x0, optimizer='gd', lr=0.01, max_iter=1000, tol=1e-6, mu=0.9):
        """
        Advanced optimization framework supporting:
        - Gradient Descent (gd)
        - SGD with minibatching
        - Momentum
        - Nesterov acceleration
        """
        x = x0.copy()
        velocity = np.zeros_like(x)
        nesterov = 'nesterov' in optimizer
        
        for i in range(max_iter):
            if nesterov:
                x_ahead = x + mu * velocity
            else:
                x_ahead = x
                
            g = self.grad(x_ahead, batch_size=32 if 'sgd' in optimizer else None)
            velocity = mu * velocity - lr * g
            x += velocity
            
            # Debugging: Print gradient norm periodically
            if i % 100 == 0:
                print(f"Iter {i}: Gradient Norm = {np.linalg.norm(g)}")
            
            # Check convergence
            if np.linalg.norm(g) < tol:
                print(f"Converged at iteration {i}")
                break
                
        return x
    
    def theoretical_limit(self, x0):
        """Compute theoretical x_infinity """
        x_mindist = self.P @ np.linalg.lstsq(self.A, self.b, rcond=None)[0]
        return x_mindist + self.P_orth @ x0
    
    def minimal_norm_solution(self):
        """Compute minimum norm solution using SVD"""
        return self.Vh.T @ np.diag(1/self.s) @ self.U.T @ self.b
    
    def validate_proof_steps(self, x0, x_final):
        """
        Empirical validation of proof components:
        1. Verify x_infinity ∈ {x | Ax = b}
        2. Check x_infinity decomposition matches theoretical guarantee 
        3. Confirm minimal distance property
        """
        # Part 1: Solution validity
        assert np.allclose(self.A @ x_final, self.b, atol=1e-4), "Final solution not in feasible set"
        
        # Part 2: Subspace decomposition
        theoretical = self.theoretical_limit(x0)
        assert np.allclose(x_final, theoretical, atol=1e-4), "Subspace decomposition mismatch"
        
        # Part 3: Minimal distance property
        res = minimize(lambda x: np.linalg.norm(x - x0)**2,
                       x0=x0,
                       constraints={'type': 'eq', 'fun': lambda x: self.A @ x - self.b})
        assert np.allclose(x_final, res.x, atol=1e-4), "Violation of minimal distance property"
        
        print("All proof steps validated successfully!")

    def visualization_engine(self, x0, x_final):
        """Advanced visualization of solution space and trajectories"""
        fig = plt.figure(figsize=(15, 8))
        
        # 3D visualization of critical subspaces
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_3d_subspaces(ax1, x0, x_final)
        
        # Convergence trajectory analysis
        ax2 = fig.add_subplot(122)
        self._plot_trajectory_components(ax2, x0, x_final)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_3d_subspaces(self, ax, x0, x_final):
        """3D visualization of solution space geometry"""
        # Generate basis vectors for visualization
        basis_A = orth(self.A.T).T[:3]
        basis_orth = orth(self.P_orth).T[:3]
        
        # Plot subspaces
        X, Y = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
        for b in basis_A:
            ax.plot_surface(X*b[0], Y*b[1], X*b[2], alpha=0.2, color='blue')
        for b in basis_orth:
            ax.plot_surface(X*b[0], Y*b[1], X*b[2], alpha=0.2, color='red')
            
        # Plot solutions
        ax.scatter(*x0[:3], c='k', marker='o', s=100, label='Initialization')
        ax.scatter(*x_final[:3], c='g', marker='*', s=200, label='Final Solution')
        ax.legend()
    
    def _plot_trajectory_components(self, ax, x0, x_final):
        """Plot evolution of solution components during optimization"""
        # Implement actual trajectory tracking
        components = {
            'Row Space Component': [],
            'Orthogonal Component': []
        }
        x = x0.copy()
        for _ in range(1000):
            x = x - 0.1 * self.grad(x)
            components['Row Space Component'].append(np.linalg.norm(self.P @ x))
            components['Orthogonal Component'].append(np.linalg.norm(self.P_orth @ x))
            
        for label, data in components.items():
            ax.plot(data, label=label)
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Component Norm')

# Advanced Usage Example
if __name__ == "__main__":
    # Initialize system with custom parameters
    analyzer = ImplicitBiasAnalyzer(n=100, p=200, loss_type='least_squares')  # Start with least squares

    # Experiment with different initializations
    x0 = np.random.randn(200) * 0.1  # Small initialization
    x_final = analyzer.optimize(x0, optimizer='momentum', lr=0.01, max_iter=5000)

    # Validate against theoretical predictions
    analyzer.validate_proof_steps(x0, x_final)

    # Generate interactive visualizations
    analyzer.visualization_engine(x0, x_final)
