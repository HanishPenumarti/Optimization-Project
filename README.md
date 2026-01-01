# 🚴📈 Optimization & Regression Project

This project implements **two major tasks**:
1️⃣ Regression modeling on the **Bike Sharing Demand** dataset  
2️⃣ Unconstrained optimization using **Newton’s Method** on a 10-dimensional design problem

The work strictly follows the assignment requirements and summarizes the full methodology, implementation, and conclusions.

---

## ✅ Question 1 – Bike Sharing Demand Regression

### 🔍 Objective
Build multiple regression models, evaluate them **only on the test set**, avoid data leakage, and select the best model purely based on performance.

---

## ⚙️ Pre-Processing
- Expanded `datetime` → hour, day, month, year  
- Removed `casual` and `registered` to prevent leakage  
- Standardized numerical features using `StandardScaler`  
- Applied **proper 80/20 train–test split**

---

## 🧠 Models Implemented
All models were trained using **Normal Equation**:
- Linear Regression (baseline)
- Polynomial Regression Degree 2
- Polynomial Regression Degree 3
- Polynomial Regression Degree 4
- Quadratic Polynomial with **interaction terms**

---

## 📊 Evaluation Metrics
- Mean Squared Error (MSE)  
- R² Score

---

## 🏆 Results Summary
| Model | MSE | R² | Notes |
|------|------|------|-------|
| Linear | 19954.53 | 0.3957 | High bias |
| Poly Deg-2 | 16406.41 | 0.5031 | Captures curvature |
| Poly Deg-3 | 14558.11 | 0.5591 | Better but riskier |
| Poly Deg-4 | 14500.42 | 0.5631 | Diminishing returns |
| **Quadratic + Interactions** | **14423.87** | **0.5631** | ⭐ Best |

### 🥇 Final Selected Model
**Quadratic Polynomial with Interactions**  
It achieved the lowest MSE & best R², capturing meaningful nonlinear + cross-feature effects while maintaining good bias–variance balance. :contentReference[oaicite:0]{index=0}

---

# 🌞 Question 2 – Newton’s Method Optimization

### 🎯 Problem
Minimize a strictly convex function
\[
f(x)=\sum_{i=1}^{10} ((x_i-c_i)^4 + (x_i-c_i)^2 + q_i x_i^2)
\]
representing a **solar power plant design optimization**.

---

## 🧮 Method
- Derived **analytical gradient and Hessian**
- Hessian is always positive definite → function is strictly convex
- Used full Newton steps:
\[
x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1}\nabla f(x_k)
\]
- Start point: \( x_0 = 0\)
- Stop when \(||\nabla f||^2 < 10^{-13}\)

---

## 📌 Numerical Results
- Initial objective: **≈ 1.05 × 10⁹**
- Final optimal point:
```
[24.0913, 177.4556, 1.7624, 4.6343, 0.8573,
 0.6239, 1.3678, 11.2323, 0.7694, 0.4547]
```
- Final objective: **≈ 6435.56**
- Converged in **14 iterations**
- Exhibited **quadratic convergence**

### ✔️ Conclusion
- Function is strictly convex → **unique global minimizer**
- Diagonal Hessian makes computation efficient
- Newton’s Method converges extremely fast near optimum

---

## 👨‍💻 Authors
- Kulkarni Keyur – BT2024025  
- Penumarti Hanish – BT2024190  
- Shashank Peddi – BT2024210  

---

## 🤝 Contribution
Feel free to fork, explore, and extend!
