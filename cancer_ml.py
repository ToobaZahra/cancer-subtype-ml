import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved!")

# Feature importance
importances = pd.Series(model.feature_importances_, index=data.feature_names)
top10 = importances.nlargest(10)
plt.figure(figsize=(8,5))
top10.plot(kind='barh')
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved!")

# PCA plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X))
plt.figure(figsize=(7,5))
for label, name in enumerate(data.target_names):
    plt.scatter(X_pca[y==label, 0], X_pca[y==label, 1], label=name, alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.savefig('pca_plot.png')
print("PCA plot saved!")