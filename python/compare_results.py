import numpy as np
import pandas as pd

# -----------------------------
# 1) LOAD DATA FROM FILES
# -----------------------------
weights = np.loadtxt("../data/weight_matrix.csv", delimiter=",")
bias = np.loadtxt("../data/bias_vector.csv", delimiter=",")
X_test = np.loadtxt("../data/X_test.csv", delimiter=",")

# -----------------------------
# 2) PLAIN SIGMOID FUNCTION
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def poly_sigmoid(z, a0=0.5, a1=0.25, a2=-0.015):
    return a0 + a1 * z + a2 * (z**2)

# -----------------------------
# 3) COMPUTE RESULTS
# -----------------------------
plain_results = []
for x in X_test:
    z = np.dot(weights, x) + bias
    exact = sigmoid(z)
    poly = poly_sigmoid(z)
    plain_results.append((z, exact, poly))

plain_df = pd.DataFrame(plain_results, columns=["z", "exact_sigmoid", "poly_sigmoid"])
plain_df.to_csv("../data/plain_results.csv", index=False)

print("Plain results saved to ../data/plain_results.csv")