"""
NON-LINEAR SVM (kernel trick) on the Kaggle "Social Network Ads" dataset
=========================================================================
Dataset : https://www.kaggle.com/datasets/rakeshrau/social-network-ads
File    : Social_Network_Ads.csv   (400 rows)
Columns : User ID, Gender, Age, EstimatedSalary, Purchased (target 0/1)

Why this dataset for NON-LINEAR SVM?
    Only 2 useful features (Age, EstimatedSalary) and the "Purchased" class
    forms a curved / clustered region, so a straight line cannot separate it
    well. RBF kernel gives a clear improvement over the linear kernel, and
    the 2-D decision boundary can be plotted.

How to get the data (pick one):
    1. Download Social_Network_Ads.csv from the Kaggle link above and put it
       next to this script (or change CSV_PATH below).
    2. Or:  pip install kagglehub   (script will download it automatically;
       needs a Kaggle account / API token).

Run:
    pip install numpy pandas scikit-learn matplotlib
    python nonlinear_svm.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
CSV_PATH = "Social_Network_Ads.csv"   # Kaggle file name
os.makedirs("plots", exist_ok=True)


def load_data(path=CSV_PATH):
    if os.path.exists(path):
        return pd.read_csv(path)
    try:  # optional automatic download
        import kagglehub
        folder = kagglehub.dataset_download("rakeshrau/social-network-ads")
        return pd.read_csv(os.path.join(folder, "Social_Network_Ads.csv"))
    except Exception as e:
        raise SystemExit(
            f"Could not find '{path}'.\n"
            "Download Social_Network_Ads.csv from "
            "https://www.kaggle.com/datasets/rakeshrau/social-network-ads "
            f"and place it next to this script.\n(kagglehub error: {e})"
        )


df = load_data()
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values:", int(df.isna().sum().sum()))
print("Class counts:\n", df["Purchased"].value_counts().rename({0: "Not purchased", 1: "Purchased"}), "\n")

X = df[["Age", "EstimatedSalary"]].values     # 2 features -> easy to plot
y = df["Purchased"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

# Scale: Age (~20-60) and Salary (~15k-150k) are on very different scales
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. COMPARE KERNELS
# ----------------------------------------------------------------------
kernels = {
    "Linear":              SVC(kernel="linear", C=1.0),
    "Polynomial (deg=3)":  SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "RBF (Gaussian)":      SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Sigmoid":             SVC(kernel="sigmoid", C=1.0, gamma="scale"),
}

print("=" * 55)
print("KERNEL COMPARISON")
print("=" * 55)
results = {}
for name, clf in kernels.items():
    clf.fit(X_train_s, y_train)
    tr = clf.score(X_train_s, y_train)
    te = clf.score(X_test_s, y_test)
    results[name] = te
    print(f"{name:<20} train={tr:.4f}   test={te:.4f}   SVs={len(clf.support_vectors_)}")

# ----------------------------------------------------------------------
# 4. TUNE THE RBF KERNEL (C and gamma)
# ----------------------------------------------------------------------
param_grid = {"C": [0.1, 1, 10, 100],
              "gamma": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_s, y_train)
best = grid.best_estimator_

y_pred = best.predict(X_test_s)
print("\n" + "=" * 55)
print("TUNED RBF SVM")
print("=" * 55)
print("Best parameters :", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.4f}")
print(f"Test accuracy   : {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Not purchased", "Purchased"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Not purchased", "Purchased"]).plot(cmap="Blues")
plt.title("Tuned RBF SVM - Confusion Matrix")
plt.savefig("plots/nonlinear_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 5. DECISION BOUNDARY PLOTS (drawn in original Age / Salary units)
# ----------------------------------------------------------------------
def plot_boundary(clf, ax, title):
    a_min, a_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    s_min, s_max = X[:, 1].min() - 10000, X[:, 1].max() + 10000
    aa, ss = np.meshgrid(np.linspace(a_min, a_max, 300),
                         np.linspace(s_min, s_max, 300))
    grid_pts = scaler.transform(np.c_[aa.ravel(), ss.ravel()])
    Z = clf.predict(grid_pts).reshape(aa.shape)

    ax.contourf(aa, ss, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm",
               edgecolors="k", s=25)
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
plot_boundary(kernels["Linear"], axes[0], f"Linear  (test acc={results['Linear']:.2f})")
plot_boundary(kernels["RBF (Gaussian)"], axes[1],
              f"RBF default  (test acc={results['RBF (Gaussian)']:.2f})")
plot_boundary(best, axes[2],
              f"RBF tuned {grid.best_params_}\n(test acc={accuracy_score(y_test, y_pred):.2f})")
plt.tight_layout()
plt.savefig("plots/nonlinear_boundaries.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. PREDICT FOR A NEW USER
# ----------------------------------------------------------------------
new_user = np.array([[30, 87000]])   # Age=30, Salary=87,000
pred = best.predict(scaler.transform(new_user))[0]
print("\nPrediction for Age=30, Salary=87000 ->",
      "Will PURCHASE" if pred == 1 else "Will NOT purchase")
print("Plots saved in ./plots/")
