# Step 1 — Import libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 2 — Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Step 3 — Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 — Define KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.bincount(k_nearest_labels).argmax()

# Step 5 — Train model
clf = KNN(k=3)
clf.fit(X_train, y_train)

# Step 6 — Test model
y_pred = clf.predict(X_test)

# Step 7 — Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Step 8 — Visualization
n_samples = 10
X_vis = X_test[:n_samples]
y_vis = y_test[:n_samples]
y_pred_vis = clf.predict(X_vis)

plt.figure(figsize=(12, 4))
for i in range(n_samples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_vis[i].reshape(8, 8), cmap="gray")
    plt.title(f"True: {y_vis[i]}\nPred: {y_pred_vis[i]}")
    plt.axis("off")

plt.suptitle("k-NN Predictions vs Actual Digits", fontsize=14)
plt.savefig("results.png")
plt.show()


# Step 9 — Compare with scikit-learn's KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

# Train scikit-learn k-NN
sk_clf = KNeighborsClassifier(n_neighbors=3)
sk_clf.fit(X_train, y_train)

# Test
sk_accuracy = sk_clf.score(X_test, y_test)

print(f"Scratch k-NN Accuracy: {accuracy:.2f}")
print(f"scikit-learn k-NN Accuracy: {sk_accuracy:.2f}")
