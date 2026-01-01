import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load data
train = pd.read_csv("train.csv")


def preprocess_datetime(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    return df.drop(columns=["datetime"])


train = preprocess_datetime(train)

# Prepare X, y
target = "count"
drop_cols = ["casual", "registered"]
X = train.drop(columns=[target] + drop_cols)
y = train[target].values

# List of Features
numeric_features = X.columns.tolist()


# Train-validation split
def split(X, y, test_size=0.2):
    np.random.seed(42)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_val = int(n * test_size)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val


X_train, X_val, y_train, y_val = split(X, y, test_size=0.2)


# Preprocessing with sklearn
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_features)]
)

# Scale features
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)


# Helper functions
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def train_normal_eq(X, y):
    # θ = (X^T X)^(-1) X^T y
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def mse(y_true, y_pred):
    diff = y_true - y_pred
    return np.mean(diff**2)


def r2_score(y_true, y_pred):
    result = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - result / total


def evaluate(X, y, weight):
    pred = X @ weight
    return mse(y, pred), r2_score(y, pred), pred


# Add bias for linear model
X_train_lr = add_bias(X_train_scaled)
X_val_lr = add_bias(X_val_scaled)

weight_linear = train_normal_eq(X_train_lr, y_train)
lr_mse, lr_r2, lr_pred = evaluate(X_val_lr, y_val, weight_linear)

print("Linear Regression:")
print("MSE:", lr_mse)
print("R²:", lr_r2)


# Polynomial expansion
def poly_expand(X, degree, interactions=False):
    features = [X]  # degree 1 always

    # degree >= 2
    if degree >= 2:
        features.append(X**2)

    # degree >= 3
    if degree >= 3:
        features.append(X**3)

    # degree >= 4
    if degree >= 4:
        features.append(X**4)

    # Quadratic interactions
    if interactions:
        interaction_terms = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                col_i = X[:, i]
                col_j = X[:, j]
                interaction_terms.append((col_i * col_j).reshape(-1, 1))
        interaction_terms = np.hstack(interaction_terms)
        features.append(interaction_terms)

    return np.hstack(features)


models = {}

# Degree 2, 3, 4 without interactions
for d in [2, 3, 4]:
    X_train_poly = poly_expand(X_train_scaled, d, interactions=False)
    X_val_poly = poly_expand(X_val_scaled, d, interactions=False)

    # Add bias after expansion
    X_train_poly_b = add_bias(X_train_poly)
    X_val_poly_b = add_bias(X_val_poly)

    weight_poly = train_normal_eq(X_train_poly_b, y_train)
    mse_d, r2_d, a = evaluate(X_val_poly_b, y_val, weight_poly)

    models[f"poly_deg_{d}"] = (mse_d, r2_d, weight_poly)

    print(f"\nPolynomial Degree {d}")
    print("MSE:", mse_d)
    print("R²:", r2_d)


# Quadratic with interaction terms
X_train_quad_int = poly_expand(X_train_scaled, 2, interactions=True)
X_val_quad_int = poly_expand(X_val_scaled, 2, interactions=True)

X_train_quad_int_b = add_bias(X_train_quad_int)
X_val_quad_int_b = add_bias(X_val_quad_int)

weight_quad_int = train_normal_eq(X_train_quad_int_b, y_train)
mse_quad, r2_quad, a = evaluate(X_val_quad_int_b, y_val, weight_quad_int)

print("\nQuadratic with Interactions")
print("MSE:", mse_quad)
print("R²:", r2_quad)


all_results = {
    "linear": (lr_mse, lr_r2, weight_linear),
    **models,  # poly_deg_2, poly_deg_3, poly_deg_4
    "quadratic_inter": (mse_quad, r2_quad, weight_quad_int),
}

best_model_name = None
best_model_metrics = None
best_mse = float("inf")

for name, (mse_val, r2_val, weight_val) in all_results.items():
    if mse_val < best_mse:
        best_mse = mse_val
        best_model_name = name
        best_model_metrics = (mse_val, r2_val, weight_val)

print("\nBEST MODEL:", best_model_name)
print("MSE:", best_model_metrics[0])
print("R²:", best_model_metrics[1])
