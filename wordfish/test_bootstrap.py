"""Test bootstrap functionality."""
import numpy as np

import warnings
warnings.simplefilter('ignore')

from wordfish import wordfish

# Set random seed for reproducibility
np.random.seed(42)

# Create sample word-document matrix
n_docs, n_words = 5, 10
true_positions = np.array([-2, -1, 0, 1, 2])
true_betas = np.random.normal(0, 1, n_words)

word_doc_matrix = np.zeros((n_docs, n_words))
for i in range(n_docs):
    for j in range(n_words):
        lambda_ij = np.exp(0.5 + true_positions[i] * true_betas[j])
        word_doc_matrix[i, j] = np.random.poisson(lambda_ij)

print("Running WORDFISH with bootstrap (10 simulations for testing)...")
results = wordfish(word_doc_matrix, verbose=True, bootstrap=True, n_bootstrap=10)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nDocument positions with 95% confidence intervals:")
print(f"{'Document':<15} {'Omega':<10} {'95% CI Lower':<15} {'95% CI Upper':<15}")
print("-" * 60)
for i, name in enumerate(results.doc_names):
    omega = results.documents[i, 0]
    if results.ci_documents is not None:
        lb = results.ci_documents[i, 0]
        ub = results.ci_documents[i, 1]
        print(f"{name:<15} {omega:>9.3f} {lb:>14.3f} {ub:>14.3f}")
    else:
        print(f"{name:<15} {omega:>9.3f} {'N/A':>14} {'N/A':>14}")

print("\nBootstrap completed successfully!")
