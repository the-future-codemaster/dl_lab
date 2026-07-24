pip install ucimlrepo

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# ==========================================
# Global Matplotlib Aesthetic Configuration
# ==========================================
# Configuring to Times New Roman, 13 Font Size, and Bold Elements
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'

# Custom dictionary for bolding legends
legend_font = {'family': 'Times New Roman', 'size': 13, 'weight': 'bold'}

# ==========================================
# Task 1: Dataset Exploration
# ==========================================
print("--- TASK 1: DATASET EXPLORATION ---\n")

# Load the dataset from the UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
features = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
columns = features + ['Class']
df = pd.read_csv(url, header=None, names=columns)

print("1. First Five Samples:")
print(df.head(), "\n")

print("2. Dataset Dimensions:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("3. Missing Values:")
print(df.isnull().sum(), "\n")

print("4. Descriptive Statistics:")
print(df.describe().T, "\n")


# ==========================================
# Task 2: Exploratory Data Analysis (EDA)[cite: 3]
# ==========================================
print("--- TASK 2: EXPLORATORY DATA ANALYSIS (Exporting Plots) ---\n")

# 1. Feature Histograms (Grid Layout)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Feature Histograms', fontweight='bold', fontsize=15)
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.histplot(data=df, x=feature, hue='Class', kde=True, ax=axes[row, col], palette='viridis')
    axes[row, col].set_xlabel(feature, fontweight='bold')
    axes[row, col].set_ylabel('Count', fontweight='bold')
    if axes[row, col].get_legend():
        plt.setp(axes[row, col].get_legend().texts, fontproperties=legend_font)

plt.tight_layout()
plt.savefig('Feature_Histograms.eps', format='eps', dpi=600)
plt.close()
print("Saved: Feature_Histograms.eps")

# 2. Boxplots (Grid Layout)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Feature Boxplots for Outlier Detection', fontweight='bold', fontsize=15)
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.boxplot(data=df, y=feature, x='Class', ax=axes[row, col], palette='Set2')
    axes[row, col].set_xlabel('Class', fontweight='bold')
    axes[row, col].set_ylabel(feature, fontweight='bold')

plt.tight_layout()
plt.savefig('Feature_Boxplots.eps', format='eps', dpi=600)
plt.close()
print("Saved: Feature_Boxplots.eps")

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
            annot_kws={'weight': 'bold', 'size': 11})
plt.title('Feature Correlation Heatmap', fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.eps', format='eps', dpi=600)
plt.close()
print("Saved: Correlation_Heatmap.eps")

# 4. Scatter Plot (Variance vs Skewness as an example based on correlation)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Variance', y='Skewness', hue='Class', palette='deep', edgecolor='k')
plt.title('Scatter Plot: Variance vs Skewness', fontweight='bold')
plt.xlabel('Variance', fontweight='bold')
plt.ylabel('Skewness', fontweight='bold')
plt.legend(prop=legend_font)
plt.tight_layout()
plt.savefig('Scatter_Plot.eps', format='eps', dpi=600)
plt.close()
print("Saved: Scatter_Plot.eps\n")


# ==========================================
# Task 3: Data Preprocessing[cite: 3]
# ==========================================
print("--- TASK 3: DATA PREPROCESSING ---\n")

# Separate features (X) and target (y)
X = df[features]
y = df['Class']

# Normalize all numerical features using Z-score standardization[cite: 3]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

print("Features Normalized. First 3 rows of scaled features:")
print(X_scaled_df.head(3), "\n")

# Split the dataset into Training (80%) and Testing (20%)[cite: 3]
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import warnings

# Suppress specific matplotlib font warnings
warnings.filterwarnings("ignore", message="findfont: Generic family 'serif' not found because none of the following families were found: Times New Roman")

# ==========================================
# Global Matplotlib Aesthetic Configuration
# ==========================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
legend_font = {'family': 'Times New Roman', 'size': 13, 'weight': 'bold'}

# Convert pandas data to numpy arrays for efficient matrix operations
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

# ==========================================
# Task 4 & 5: Perceptron Implementation and Training
# ==========================================
print("--- TASK 4 & 5: PERCEPTRON IMPLEMENTATION AND TRAINING ---\n")

class SingleLayerPerceptron:
    """
    Object-Oriented implementation of a Single Layer Perceptron from scratch.
    """
    def __init__(self, learning_rate=0.01, epochs=50, random_state=42):
        self.eta = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def step_activation(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)

        self.errors_ = []
        self.weights_history_ = []
        self.bias_history_ = []

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.w_) + self.b_
                y_hat = self.step_activation(z)
                update = self.eta * (target - y_hat)
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
            self.weights_history_.append(self.w_.copy())
            self.bias_history_.append(self.b_)

        return self

    def predict(self, X):
        z = np.dot(X, self.w_) + self.b_
        return self.step_activation(z)

# Train the model
epochs_count = 15
learning_rate_val = 0.01
model = SingleLayerPerceptron(learning_rate=learning_rate_val, epochs=epochs_count)
model.fit(X_train_np, y_train_np)

print(f"Model trained for {epochs_count} epochs with learning rate {learning_rate_val}.\n")

# Display Epoch-wise Learning for ALL 15 epochs
print("Epoch-wise Learning Summary (Full 15 Epochs):")
print(f"{'Epoch':<8} | {'Errors':<8} | {'W1 (Var)':<10} | {'W2 (Skew)':<10} | {'Bias':<10}")
print("-" * 55)
for i in range(epochs_count):
    print(f"{i+1:<8} | {model.errors_[i]:<8} | {model.weights_history_[i][0]:<10.4f} | {model.weights_history_[i][1]:<10.4f} | {model.bias_history_[i]:<10.4f}")
print("\n")


# ==========================================
# Task 6: Model Evaluation
# ==========================================
print("--- TASK 6: MODEL EVALUATION ---\n")

y_pred = model.predict(X_test_np)
TP = np.sum((y_test_np == 1) & (y_pred == 1))
TN = np.sum((y_test_np == 0) & (y_pred == 0))
FP = np.sum((y_test_np == 0) & (y_pred == 1))
FN = np.sum((y_test_np == 1) & (y_pred == 0))
cm = np.array([[TN, FP], [FN, TP]])

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Confusion Matrix:\n[[TN: {TN}, FP: {FP}]\n [FN: {FN}, TP: {TP}]]")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_score:.4f}\n")


# ==========================================
# Task 7: Mandatory Plots
# ==========================================
print("--- TASK 7: EXPORTING MANDATORY PLOTS ---\n")

# 1. Training Error vs Epoch
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker='o', color='#D95319', linewidth=2, markersize=8)
plt.title('Training Error vs Epoch', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Misclassified Samples', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Training_Error_vs_Epoch.eps', format='eps', dpi=600)
plt.close()
print("Saved: Training_Error_vs_Epoch.eps")

# 2. Weight Evolution (Grid Layout)
weights_matrix = np.array(model.weights_history_)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Evolution of Learned Weights over Epochs', fontweight='bold', fontsize=15)
features_list = ['Variance', 'Skewness', 'Curtosis', 'Entropy']
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']

for i in range(4):
    row, col = divmod(i, 2)
    axes[row, col].plot(range(1, epochs_count + 1), weights_matrix[:, i], marker='s', color=colors[i])
    axes[row, col].set_title(f'Weight: {features_list[i]}', fontweight='bold')
    axes[row, col].set_xlabel('Epoch', fontweight='bold')
    axes[row, col].set_ylabel('Weight Value', fontweight='bold')
    axes[row, col].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Weight_Evolution.eps', format='eps', dpi=600)
plt.close()
print("Saved: Weight_Evolution.eps")
