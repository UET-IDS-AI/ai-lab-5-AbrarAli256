[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/pdEiMXXt)
# AI Stats Lab — Lasso Regularization & Overfitting

This assignment explores two important machine learning concepts:

1. **Lasso Regularization (L1 penalty)**
2. **Overfitting due to increasing polynomial model complexity**

Reference lecture:
https://kuleshov-group.github.io/aml-book/contents/lecture5-regularization.html

Dataset used: **Diabetes dataset** from scikit-learn.

---

# Repository Structure

```
AIstats_lab.py
test_AIstats_lab.py
README.md
```

---

# Q1 — Lasso Regression (L1 Regularization)

Lasso adds an **L1 penalty** to the loss:

L(θ) = MSE + λ‖θ‖₁

Where

‖θ‖₁ = sum of absolute values of parameters.

Since the L1 norm is not differentiable at zero, we use the **subgradient**:

∂|θ| = sign(θ)

### Task

Implement Lasso regression using **gradient descent**.

Function:

```python
lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000)
```

Steps:

1. Load diabetes dataset
2. Train/test split
3. Standardize features
4. Add bias column
5. Implement gradient descent with L1 penalty
6. Evaluate metrics

Return:

```
train_mse
test_mse
train_r2
test_r2
theta
```

---

# Q2 — Overfitting with Polynomial Regression

Higher-order polynomial models can **overfit training data**.

Your task is to compare models with increasing polynomial degree.

Steps:

1. Load diabetes dataset
2. Use **BMI feature only** (feature index = 2)
3. Create polynomial features from **degree 1 → max_degree**
4. Train regression using the **normal equation**
5. Compute train and test MSE

Function:

```python
polynomial_overfitting_experiment(max_degree=10)
```

Return dictionary:

```
{
"degrees": list,
"train_mse": list,
"test_mse": list
}
```

Expected behavior:

* training error decreases with degree
* test error eventually increases (overfitting)

---

# Running Tests

Install dependencies:

```bash
pip install numpy scikit-learn pytest
```

Run autograder:

```bash
pytest -q
```

---

# Rules

• Do not rename functions
• Do not change return formats
• Use numpy operations where possible
