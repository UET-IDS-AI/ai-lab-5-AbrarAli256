"""
AIstats_lab.py
Student starter file for the Regularization & Overfitting lab.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def add_bias(X):
    """Prepend a column of ones to X for the intercept term."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mse(y_true, y_pred):
    """Mean Squared Error: average of squared residuals."""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    R² Score: proportion of variance explained by the model.
    1.0 = perfect fit | 0.0 = same as predicting the mean
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ─────────────────────────────────────────
# Q1 — Lasso Regression (L1 Regularization)
# ─────────────────────────────────────────

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Lasso regression via gradient descent on the diabetes dataset.

    Loss: L(theta) = MSE + lambda * ||theta||_1
    Gradient: dL/dtheta = (2/n) X^T(X*theta - y) + lambda * sign(theta)

    Parameters
    ----------
    lambda_reg : float  -- L1 regularization strength
    lr         : float  -- learning rate
    epochs     : int    -- number of gradient descent steps

    Returns
    -------
    train_mse, test_mse, train_r2, test_r2, theta
    """

    # 1. Load Data
    data = load_diabetes()
    X, y = data.data, data.target          # X: (442, 10)  y: (442,)

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Standardise Features
    # Fit scaler ONLY on training data to prevent data leakage
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4. Add Bias Column
    X_train_b = add_bias(X_train)          # (353, 11)
    X_test_b  = add_bias(X_test)           # ( 89, 11)

    # 5. Initialise Parameters
    n_params = X_train_b.shape[1]          # 11 (bias + 10 features)
    theta    = np.zeros(n_params)
    n        = len(y_train)

    # 6. Gradient Descent with L1 Penalty
    for _ in range(epochs):

        # Forward pass
        y_pred = X_train_b @ theta
        error  = y_pred - y_train

        # MSE gradient: (2/n) * X^T * error
        grad_mse = (2.0 / n) * (X_train_b.T @ error)

        # L1 subgradient: sign(theta) -- do NOT regularise the bias (index 0)
        l1_grad    = np.sign(theta)
        l1_grad[0] = 0.0

        # Update parameters
        theta -= lr * (grad_mse + lambda_reg * l1_grad)

    # 7. Predictions & Metrics
    train_pred = X_train_b @ theta
    test_pred  = X_test_b  @ theta

    train_mse = mse(y_train, train_pred)
    test_mse  = mse(y_test,  test_pred)
    train_r2  = r2_score(y_train, train_pred)
    test_r2   = r2_score(y_test,  test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# ─────────────────────────────────────────
# Q2 — Polynomial Overfitting Experiment
# ─────────────────────────────────────────

def polynomial_overfitting_experiment(max_degree=10):
    """
    Fit polynomial regression models of increasing degree using only the
    BMI feature, and record train/test MSE to observe overfitting.

    Normal Equation (closed-form solution):
        theta = (X^T X)^{-1} X^T y  -- via pseudoinverse for stability

    Parameters
    ----------
    max_degree : int -- highest polynomial degree to evaluate

    Returns
    -------
    dict with keys:
        "degrees"   : list[int]
        "train_mse" : list[float]
        "test_mse"  : list[float]
    """

    # 1. Load Data & Select BMI Feature (index 2)
    data  = load_diabetes()
    X_bmi = data.data[:, 2].reshape(-1, 1)
    y     = data.target

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bmi, y, test_size=0.2, random_state=42
    )

    degrees      = []
    train_errors = []
    test_errors  = []

    # 3. Loop Over Polynomial Degrees
    for degree in range(1, max_degree + 1):

        # Expand: [1, x, x^2, ..., x^degree]
        poly         = PolynomialFeatures(degree=degree, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly  = poly.transform(X_test)

        # Normal Equation: theta = pinv(X^T X) * X^T y
        XtX   = X_train_poly.T @ X_train_poly
        Xty   = X_train_poly.T @ y_train
        theta = np.linalg.pinv(XtX) @ Xty

        # Evaluate on both splits
        train_pred = X_train_poly @ theta
        test_pred  = X_test_poly  @ theta

        degrees.append(degree)
        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test,  test_pred))

    return {
        "degrees"  : degrees,
        "train_mse": train_errors,
        "test_mse" : test_errors,
    }
