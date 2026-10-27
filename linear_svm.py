
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

RANDOM_STATE = 42
CSV_PATH = "data.csv"          # Kaggle file name
os.makedirs("plots", exist_ok=True)


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(path=CSV_PATH):
    if os.path.exists(path):
        return pd.read_csv(path)
    try:  # optional automatic download
        import kagglehub
        folder = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        return pd.read_csv(os.path.join(folder, "data.csv"))
    except Exception as e:
        raise SystemExit(
            f"Could not find '{path}'.\n"
            "Download data.csv from "
            "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data "
            f"and place it next to this script.\n(kagglehub error: {e})"
        )


df = load_data()
print("Shape:", df.shape)
print(df.head(), "\n")

# ----------------------------------------------------------------------
# 2. PRE-PROCESSING
# ----------------------------------------------------------------------
# Remove useless columns ('id' and the empty 'Unnamed: 32' column)
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Encode target: M (malignant) = 1, B (benign) = 0
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
print("Missing values:", int(df.isna().sum().sum()))
print("Class counts  :\n", df["diagnosis"].value_counts().rename({0: "Benign", 1: "Malignant"}), "\n")

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# ----------------------------------------------------------------------
# 3. TRAIN LINEAR SVM  (scaling is essential for SVM -> use a pipeline)
# ----------------------------------------------------------------------
model = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("=" * 55)
print("LINEAR SVM (C=1.0)")
print("=" * 55)
print(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"5-fold CV acc : {cross_val_score(model, X, y, cv=5).mean():.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title("Linear SVM - Confusion Matrix")
plt.savefig("plots/linear_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (only C matters for a linear kernel)
# ----------------------------------------------------------------------
grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(kernel="linear")),
    param_grid={"svc__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
print("Best C        :", grid.best_params_["svc__C"])
print(f"Best CV acc   : {grid.best_score_:.4f}")
print(f"Tuned test acc: {accuracy_score(y_test, grid.predict(X_test)):.4f}\n")

# ----------------------------------------------------------------------
# 5. INTERPRET THE MODEL: largest |weights| = most important features
# ----------------------------------------------------------------------
svm = model.named_steps["svc"]
weights = pd.Series(svm.coef_[0], index=X.columns)
top = weights.reindex(weights.abs().sort_values(ascending=False).index).head(10)
print("Top 10 features by |weight|:\n", top.round(3), "\n")
print("Support vectors used:", len(svm.support_vectors_), "of", len(X_train))

plt.figure(figsize=(7, 5))
top.sort_values().plot(kind="barh", color=np.where(top.sort_values() > 0, "crimson", "steelblue"))
plt.title("Linear SVM - Top 10 feature weights\n(red = pushes towards Malignant)")
plt.tight_layout()
plt.savefig("plots/linear_feature_weights.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. 2-D VISUALISATION OF THE LINEAR BOUNDARY (2 features only)
# ----------------------------------------------------------------------
f1, f2 = "radius_mean", "texture_mean"
X2 = StandardScaler().fit_transform(df[[f1, f2]])
svm2 = SVC(kernel="linear", C=1.0).fit(X2, y)

xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 300),
                     np.linspace(X2[:, 1].min() - 1, X2[:, 1].max() + 1, 300))
Z = svm2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z > 0, alpha=0.2, cmap="coolwarm")
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
plt.scatter(*svm2.support_vectors_.T, s=90, facecolors="none", edgecolors="green")
plt.xlabel(f1 + " (scaled)")
plt.ylabel(f2 + " (scaled)")
plt.title(f"Linear SVM boundary using only 2 features\n2-feature accuracy = {svm2.score(X2, y):.3f}")
plt.tight_layout()
plt.savefig("plots/linear_2d_boundary.png", dpi=150)
plt.close()

print("\nPlots saved in ./plots/")
