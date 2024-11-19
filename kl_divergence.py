import numpy as np

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler Divergence between two distributions.
    
    Parameters:
    - p (list or numpy array): The first probability distribution (input script).
    - q (list or numpy array): The second probability distribution (reference/corpus).
    
    Returns:
    - float: The KL divergence value.
    """
    # Convert to numpy arrays for element-wise operations
    p = np.array(p)
    q = np.array(q)
    
    # Ensure no division by zero and avoid log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)  # Clip values to avoid log(0)
    q = np.clip(q, epsilon, 1)  # Clip values to avoid division by zero
    
    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))
    return kl

# Example usage
p = [0.4, 0.5, 0.1]  # Input script distribution
q = [0.3, 0.4, 0.3]  # Corpus distribution

kl_result = kl_divergence(p, q)
print(f"KL Divergence: {kl_result:.5f}")
