Description

This project explores the implicit bias of gradient-based optimization methods in underdetermined regression problems. When multiple global minima exist, gradient descent and its variants converge to solutions that are closest to the initialization in Euclidean distance. This bias leads to the minimal norm solutions and characterize the role of initialization in determining the final solution.

Key Features

Theoretical Foundations: Implementation of proofs characterizing the limit iterate x_infinity and its minimal distance properties.

Advanced Optimization: Support for gradient descent, momentum, Nesterov acceleration, and stochastic gradient descent (SGD).
Loss Functions: Includes least squares and Huber loss for regression

Singular Value Decomposition (SVD) for subspace analysis
Quadratic programming for minimal distance validation

Who Is This For?
Researchers in optimization and machine learning.
Graduate students exploring implicit bias in over-parameterized models.
Practitioners interested in understanding the behavior of gradient-based methods.






Example Usage
# Initialize system with custom parameters
analyzer = ImplicitBiasAnalyzer(n=100, p=200, loss_type='least_squares')

# Run optimization with momentum
x0 = np.random.randn(200) * 0.1  # Small initialization
x_final = analyzer.optimize(x0, optimizer='momentum', lr=0.01, max_iter=5000)

# Validate theoretical predictions
analyzer.validate_proof_steps(x0, x_final)

# Visualize results
analyzer.visualization_engine(x0, x_final)


