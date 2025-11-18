""""
Group C
BSc Economics, Finance and Data Science
Assessed Problem Set 1 
05/11/2025
Problem 3
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def f(t):
    return 1/3 + 0.25 * np.sin(3 * np.pi * t) + t/4


def simulate_label(x1, x2, alpha):
    boundary = f(x1) - alpha
    indicator = 1 if x2 < boundary else 0
    prob = max(indicator, alpha)
    Y = int(np.random.rand() < prob)
    return Y


def generate_training_data(n=100, alpha=0.2):
    X = np.random.uniform(0, 1, size=(n, 2))
    y = np.array([simulate_label(X[i, 0], X[i, 1], alpha) for i in range(n)])
    return X, y


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def knn_predict(X_train, y_train, X_test, k=1, metric='euclidean'):
    if metric == 'euclidean':
        distance_fn = euclidean_distance
    elif metric == 'manhattan':
        distance_fn = manhattan_distance
    else:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
    
    n_test = X_test.shape[0]
    y_pred = np.zeros(n_test, dtype=int)
    
    for i in range(n_test):
        distances = np.array([distance_fn(X_test[i], X_train[j]) 
                             for j in range(len(X_train))])
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        y_pred[i] = np.bincount(k_nearest_labels).argmax()
    
    return y_pred


def classification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def kfold_cross_validation(D, A, loss_fn, k=5):
    X, y = D
    n = len(y)
    
    indices = np.random.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        fold_indices = indices[current:current + fold_size]
        folds.append(fold_indices)
        current += fold_size
    
    errors = np.zeros(k)
    
    for j in range(k):
        val_indices = folds[j]
        train_indices = np.concatenate([folds[i] for i in range(k) if i != j])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        f_j = A((X_train, y_train))
        y_pred = f_j(X_val)
        errors[j] = loss_fn(y_val, y_pred)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    
    return errors, mean_error, std_error

# Question 1 - Generating and viaualizing training data

n = 100
alpha = 0.2
X_train, y_train = generate_training_data(n=n, alpha=alpha)

print(f"\nGenerated training data: n={n}, α={alpha}")
print(f"Class 0: {np.sum(y_train == 0)} samples")
print(f"Class 1: {np.sum(y_train == 1)} samples")

plt.figure(figsize=(8, 6))
class_0 = y_train == 0
class_1 = y_train == 1
plt.scatter(X_train[class_0, 0], X_train[class_0, 1], 
            c='red', marker='o', s=50, alpha=0.6, label='Y = 0')
plt.scatter(X_train[class_1, 0], X_train[class_1, 1], 
            c='blue', marker='o', s=50, alpha=0.6, label='Y = 1')
plt.xlabel('X₁', fontsize=12)
plt.ylabel('X₂', fontsize=12)
plt.title(f'Training Dataset (n={n}, α={alpha})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.savefig('question1_training_data.png', dpi=150, bbox_inches='tight')

# Question 2 - 5-Fold Cross-Validation for k-NN

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
num_folds = 5

cv_means = []
cv_stds = []

for k in k_values:
    def A(D, k_param=k):
        X_tr, y_tr = D
        def predictor(X_test):
            return knn_predict(X_tr, y_tr, X_test, k=k_param)
        return predictor
    
    errors, mean_error, std_error = kfold_cross_validation(
        D=(X_train, y_train),
        A=A,
        loss_fn=classification_error,
        k=num_folds
    )
    
    cv_means.append(mean_error)
    cv_stds.append(std_error)
    
    print(f"k={k:2d}: CV Error = {mean_error:.4f} ± {std_error:.4f}")

plt.figure(figsize=(10, 6))
plt.errorbar(k_values, cv_means, yerr=cv_stds, marker='o', capsize=5, 
             linewidth=2, markersize=8, label='5-fold CV error')
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Classification Error Rate', fontsize=12)
plt.title('5-Fold Cross-Validation Error vs k for k-NN Classifier', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_values)
plt.tight_layout()
plt.show()
plt.savefig('question2_cv_results.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to 'question2_cv_results.png'")
print(f"Best k: {k_values[np.argmin(cv_means)]} (CV error = {min(cv_means):.4f})\n")


# Question 3 - Repeating Cross-Validation with Different Random Seed

np.random.seed(0)

cv_means_2 = []
cv_stds_2 = []

for k in k_values:
    def A(D, k_param=k):
        X_tr, y_tr = D
        def predictor(X_test):
            return knn_predict(X_tr, y_tr, X_test, k=k_param)
        return predictor
    
    errors, mean_error, std_error = kfold_cross_validation(
        D=(X_train, y_train),
        A=A,
        loss_fn=classification_error,
        k=num_folds
    )
    
    cv_means_2.append(mean_error)
    cv_stds_2.append(std_error)
    
    print(f"k={k:2d}: CV Error = {mean_error:.4f} ± {std_error:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.errorbar(k_values, cv_means, yerr=cv_stds, marker='o', capsize=5, 
             linewidth=2, markersize=8, label='Run 1', color='blue')
ax1.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax1.set_ylabel('Classification Error Rate', fontsize=12)
ax1.set_title('Run 1: 5-Fold CV Error vs k', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks(k_values)

ax2.errorbar(k_values, cv_means_2, yerr=cv_stds_2, marker='s', capsize=5, 
             linewidth=2, markersize=8, label='Run 2', color='red')
ax2.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax2.set_ylabel('Classification Error Rate', fontsize=12)
ax2.set_title('Run 2: 5-Fold CV Error vs k', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(k_values)

plt.tight_layout()
plt.show()
plt.savefig('question3_comparison.png', dpi=150, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.errorbar(k_values, cv_means, yerr=cv_stds, marker='o', capsize=5, 
             linewidth=2, markersize=8, label='Run 1', alpha=0.7)
plt.errorbar(k_values, cv_means_2, yerr=cv_stds_2, marker='s', capsize=5, 
             linewidth=2, markersize=8, label='Run 2', alpha=0.7)
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Classification Error Rate', fontsize=12)
plt.title('Comparison: Two Runs of 5-Fold Cross-Validation', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_values)
plt.tight_layout()
plt.show()
plt.savefig('question3_overlay.png', dpi=150, bbox_inches='tight')

print(f"\nPlots saved to 'question3_comparison.png' and 'question3_overlay.png'")
print(f"Run 1 best k: {k_values[np.argmin(cv_means)]} (error = {min(cv_means):.4f})")
print(f"Run 2 best k: {k_values[np.argmin(cv_means_2)]} (error = {min(cv_means_2):.4f})")

correlation = np.corrcoef(cv_means, cv_means_2)[0, 1]
print(f"Correlation between runs: {correlation:.4f}\n")

# Question 4: Best k and Decision Boundary Visualization

np.random.seed(42)
X_train, y_train = generate_training_data(n=100, alpha=0.2)

cv_means = []
for k in k_values:
    def A(D, k_param=k):
        X_tr, y_tr = D
        def predictor(X_test):
            return knn_predict(X_tr, y_tr, X_test, k=k_param)
        return predictor
    
    errors, mean_error, std_error = kfold_cross_validation(
        D=(X_train, y_train),
        A=A,
        loss_fn=classification_error,
        k=num_folds
    )
    cv_means.append(mean_error)

best_k = k_values[np.argmin(cv_means)]
print(f"\nBest k = {best_k} (CV error = {min(cv_means):.4f})")

x1_grid = np.linspace(0, 1, 200)
x2_grid = np.linspace(0, 1, 200)
X1_mesh, X2_mesh = np.meshgrid(x1_grid, x2_grid)
X_test_grid = np.c_[X1_mesh.ravel(), X2_mesh.ravel()]

y_pred_grid = knn_predict(X_train, y_train, X_test_grid, k=best_k)
y_pred_mesh = y_pred_grid.reshape(X1_mesh.shape)

plt.figure(figsize=(10, 8))
plt.contourf(X1_mesh, X2_mesh, y_pred_mesh, levels=1, alpha=0.3, colors=['red', 'blue'])
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
            c='red', marker='o', s=50, edgecolors='black', label='Y = 0', alpha=0.7)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
            c='blue', marker='o', s=50, edgecolors='black', label='Y = 1', alpha=0.7)
x1_boundary = np.linspace(0, 1, 200)
x2_boundary = f(x1_boundary) - alpha
plt.plot(x1_boundary, x2_boundary, 'g-', linewidth=2, label='True Boundary')
plt.xlabel('X₁', fontsize=12)
plt.ylabel('X₂', fontsize=12)
plt.title(f'k-NN Decision Boundary (k={best_k}, α={alpha})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.savefig('question4_decision_boundary.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to 'question4_decision_boundary.png'\n")

# Question 5: Effect of Noise Level α

alpha_values = [0, 0.1, 0.2, 0.3, 0.4]
best_k_per_alpha = []
all_cv_results = {}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, alpha_val in enumerate(alpha_values):
    np.random.seed(42)
    X_train_alpha, y_train_alpha = generate_training_data(n=100, alpha=alpha_val)
    
    cv_means_alpha = []
    cv_stds_alpha = []
    
    for k in k_values:
        def A(D, k_param=k):
            X_tr, y_tr = D
            def predictor(X_test):
                return knn_predict(X_tr, y_tr, X_test, k=k_param)
            return predictor
        
        np.random.seed(42)
        errors, mean_error, std_error = kfold_cross_validation(
            D=(X_train_alpha, y_train_alpha),
            A=A,
            loss_fn=classification_error,
            k=num_folds
        )
        
        cv_means_alpha.append(mean_error)
        cv_stds_alpha.append(std_error)
    
    best_k_alpha = k_values[np.argmin(cv_means_alpha)]
    best_k_per_alpha.append(best_k_alpha)
    all_cv_results[alpha_val] = (cv_means_alpha, cv_stds_alpha)
    
    print(f"α = {alpha_val:.1f}: Best k = {best_k_alpha} (CV error = {min(cv_means_alpha):.4f})")
    
    axes[idx].errorbar(k_values, cv_means_alpha, yerr=cv_stds_alpha, 
                       marker='o', capsize=5, linewidth=2, markersize=8)
    axes[idx].set_xlabel('k', fontsize=11)
    axes[idx].set_ylabel('CV Error', fontsize=11)
    axes[idx].set_title(f'α = {alpha_val:.1f}, Best k = {best_k_alpha}', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(k_values)

axes[-1].axis('off')
plt.tight_layout()
plt.show()
plt.savefig('question5_alpha_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to 'question5_alpha_comparison.png'")

fig, axes = plt.subplots(1, len(alpha_values), figsize=(20, 4))

for idx, alpha_val in enumerate(alpha_values):
    np.random.seed(42)
    X_train_alpha, y_train_alpha = generate_training_data(n=100, alpha=alpha_val)
    best_k_alpha = best_k_per_alpha[idx]
    
    y_pred_grid = knn_predict(X_train_alpha, y_train_alpha, X_test_grid, k=best_k_alpha)
    y_pred_mesh = y_pred_grid.reshape(X1_mesh.shape)
    
    axes[idx].contourf(X1_mesh, X2_mesh, y_pred_mesh, levels=1, alpha=0.3, colors=['red', 'blue'])
    axes[idx].scatter(X_train_alpha[y_train_alpha==0, 0], X_train_alpha[y_train_alpha==0, 1], 
                      c='red', marker='o', s=30, edgecolors='black', alpha=0.6)
    axes[idx].scatter(X_train_alpha[y_train_alpha==1, 0], X_train_alpha[y_train_alpha==1, 1], 
                      c='blue', marker='o', s=30, edgecolors='black', alpha=0.6)
    x1_boundary = np.linspace(0, 1, 200)
    x2_boundary = f(x1_boundary) - alpha_val
    axes[idx].plot(x1_boundary, x2_boundary, 'g-', linewidth=2)
    axes[idx].set_xlabel('X₁', fontsize=11)
    axes[idx].set_ylabel('X₂', fontsize=11)
    axes[idx].set_title(f'α={alpha_val:.1f}, k={best_k_alpha}', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, 1)
    axes[idx].set_ylim(0, 1)

plt.tight_layout()
plt.show()
plt.savefig('question5_boundaries.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to 'question5_boundaries.png'\n")

# Question 6: Multiple Hyperparameter Tuning

np.random.seed(42)
X_train, y_train = generate_training_data(n=100, alpha=0.2)

k_values_q6 = [1, 3, 5, 7, 9]
metrics = ['euclidean', 'manhattan']

results = []

for metric in metrics:
    for k in k_values_q6:
        def A(D, k_param=k, metric_param=metric):
            X_tr, y_tr = D
            def predictor(X_test):
                return knn_predict(X_tr, y_tr, X_test, k=k_param, metric=metric_param)
            return predictor
        
        np.random.seed(42)
        errors, mean_error, std_error = kfold_cross_validation(
            D=(X_train, y_train),
            A=A,
            loss_fn=classification_error,
            k=num_folds
        )
        
        results.append({
            'k': k,
            'metric': metric,
            'cv_error': mean_error,
            'cv_std': std_error
        })
        
        print(f"k={k}, metric={metric:10s}: CV Error = {mean_error:.4f} ± {std_error:.4f}")

results_sorted = sorted(results, key=lambda x: x['cv_error'])
best_config = results_sorted[0]

print("\n" + "=" * 60)
print(f"Best Configuration: k={best_config['k']}, metric={best_config['metric']}")
print(f"CV Error: {best_config['cv_error']:.4f} ± {best_config['cv_std']:.4f}")

euclidean_errors = [r['cv_error'] for r in results if r['metric'] == 'euclidean']
manhattan_errors = [r['cv_error'] for r in results if r['metric'] == 'manhattan']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for metric in metrics:
    metric_results = [r for r in results if r['metric'] == metric]
    k_vals = [r['k'] for r in metric_results]
    cv_errs = [r['cv_error'] for r in metric_results]
    cv_stds = [r['cv_std'] for r in metric_results]
    
    ax1.errorbar(k_vals, cv_errs, yerr=cv_stds, marker='o' if metric=='euclidean' else 's',
                 capsize=5, linewidth=2, markersize=8, label=metric.capitalize(), alpha=0.7)

ax1.set_xlabel('k (Number of Neighbors)', fontsize=12)
ax1.set_ylabel('CV Error Rate', fontsize=12)
ax1.set_title('CV Error vs k for Different Metrics', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(k_values_q6)

metric_means = [np.mean(euclidean_errors), np.mean(manhattan_errors)]
metric_stds = [np.std(euclidean_errors), np.std(manhattan_errors)]

x_pos = np.arange(len(metrics))
ax2.bar(x_pos, metric_means, yerr=metric_stds, capsize=5, 
        color=['blue', 'orange'], alpha=0.7)
ax2.set_ylabel('Average CV Error', fontsize=12)
ax2.set_title('Average Performance by Metric', fontsize=13)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([m.capitalize() for m in metrics])
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.savefig('question6_hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to 'question6_hyperparameter_tuning.png'")

k_effect = {}
for k in k_values_q6:
    k_errors = [r['cv_error'] for r in results if r['k'] == k]
    k_effect[k] = np.std(k_errors)

metric_effect = {}
for metric in metrics:
    metric_errors = [r['cv_error'] for r in results if r['metric'] == metric]
    metric_effect[metric] = np.std(metric_errors)

avg_k_variance = np.mean(list(k_effect.values()))
avg_metric_variance = np.mean(list(metric_effect.values()))

print("Parameter Importance Analysis:")
print(f"Average variance across k values: {avg_k_variance:.6f}")
print(f"Average variance across metrics: {avg_metric_variance:.6f}")

if avg_k_variance > avg_metric_variance:
    print("Choice of k appears more important than choice of metric")
else:
    print("Choice of metric appears more important than choice of k")