# Mathematics for Machine Learning (MML) — Comprehensive Computational Solutions Suite

<p align="center">
  <h2 align="center">Mathematics for Machine Learning (Deisenroth, Faisal & Ong)</h2>
  <p align="center">
    <strong>Complete Computational Implementations & Mathematical Derivations across Chapters 2 through 12</strong><br>
    <em>MSc Artificial Intelligence — Coursework in Mathematics for AI</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Textbook-Cambridge_University_Press-8A2BE2?style=for-the-badge" alt="Textbook"/>
    <img src="https://img.shields.io/badge/Chapters-2_to_12_(Complete)-blue?style=for-the-badge" alt="Chapters"/>
    <img src="https://img.shields.io/badge/Total_Cells-251_Cells-green?style=for-the-badge" alt="Cells"/>
    <img src="https://img.shields.io/badge/Concurrency-Multi--Threaded_Parallel-orange?style=for-the-badge" alt="Concurrency"/>
    <img src="https://img.shields.io/badge/Notebooks-100%25_Standalone-brightgreen?style=for-the-badge" alt="Standalone"/>
    <a href="MML_Book_Summary_Cheatsheet.md"><img src="https://img.shields.io/badge/Cheatsheet-MML_Book_Summary-purple?style=for-the-badge" alt="Cheatsheet"/></a>
  </p>
</p>

---

## 📖 Source Material & Attribution

> **Note on Textbook Source Material:**  
> All mathematical formulations, exercise problem statements, theoretical questions, and computational models across this suite are drawn directly from the authoritative textbook:  
> **"Mathematics for Machine Learning"** by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (Cambridge University Press, 2020).  
> *Official textbook website: [mml-book.com](https://mml-book.com/)*  
>
> In accordance with academic integrity and copyright policies, the textbook PDF is excluded from version control via `.gitignore`. Every notebook includes full bibliographic attribution in its markdown header.

---

## 🧭 Overview & Architectural Design

> 💡 **Comprehensive Theoretical Reference & Cheatsheet:**  
> For an exhaustive, chapter-by-chapter conceptual summary, key definitions, theorems, matrix calculus identities, and algorithmic steps across all 12 chapters, see the companion [**MML Book Summary & Mathematical Cheatsheet**](MML_Book_Summary_Cheatsheet.md).

This directory provides an exhaustive, runnable computational companion solving exercises and practical implementations across both parts of the textbook:

1. **Part I: Mathematical Foundations (Chapters 2–7)**: Complete, uncompressed solutions for all **75 numbered end-of-chapter exercises** covering linear algebra, analytic geometry, matrix factorisations, multivariate calculus, probability theory, and continuous optimisation.
2. **Part II: Central Machine Learning Algorithms (Chapters 8–12)**: Comprehensive formulations covering all sections and official tutorial problem sets for model fitting, linear regression, dimensionality reduction (PCA), density estimation (GMMs), and classification (SVMs).

### ⚡ Standalone Parallel Architecture
To take full advantage of multi-core CPU architectures (**16 CPU cores**, **~31 GB RAM**) while keeping each notebook **100% self-contained and standalone** with zero external file dependencies:
- All parallel worker routines are defined directly in-cell with `concurrent.futures.ThreadPoolExecutor`.
- The Global Interpreter Lock (GIL) is automatically released during NumPy, SciPy (SLSQP, SVD, linear solvers), and BLAS/LAPACK operations, delivering high multi-core utilization without Python `pickle` serialization constraints.
- **Bias-Variance Monte Carlo:** 10,000 independent polynomial fits distributed across cores in Chapter 8.
- **Bayesian Predictive Surfaces:** Parallel grid evaluations for posterior predictive mean and epistemic variance in Chapter 9.
- **High-Dimensional Dual PCA:** Benchmark scaling up to $D = 12,000$ dimensions, demonstrating speedups exceeding **980x to 20,000x** over primal eigendecomposition in Chapter 10.
- **GMM EM Restarts:** 56 independent EM initializations executed concurrently across 16 cores for robust AIC/BIC model selection in Chapter 11.
- **SVM Hyperparameter Grid Search:** 120 cross-validation folds evaluated in parallel across 24 $(C, \gamma)$ candidate pairs in Chapter 12.

---

## 📚 Notebook Catalog (11 Chapters, 251 Cells)

### Part I: Mathematical Foundations

| Notebook | Topic | Exercise Coverage | Cells | Key Concepts & Methods |
|:---|:---|:---:|:---:|:---|
| [Ch02_Linear_Algebra.ipynb](Ch02_Linear_Algebra.ipynb) | Linear Algebra | **Ex 2.1 – 2.20** (All 20) | 50 | Gaussian elimination, vector spaces, linear independence, basis transformations, rank, kernel, matrix inverses. |
| [Ch03_Analytic_Geometry.ipynb](Ch03_Analytic_Geometry.ipynb) | Analytic Geometry | **Ex 3.1 – 3.10** (All 10) | 27 | Inner products, norms ($L_1, L_2, L_\infty$), Cauchy-Schwarz inequality, orthogonal projections, Gram-Schmidt orthogonalisation, rotation matrices. |
| [Ch04_Matrix_Decompositions.ipynb](Ch04_Matrix_Decompositions.ipynb) | Matrix Decompositions | **Ex 4.1 – 4.12** (All 12) | 29 | Determinants, traces, characteristic polynomials, eigenvalues/eigenvectors, Cholesky factorization, symmetric spectral theorem, Singular Value Decomposition (SVD). |
| [Ch05_Vector_Calculus.ipynb](Ch05_Vector_Calculus.ipynb) | Vector Calculus | **Ex 5.1 – 5.9** (All 9) | 22 | Gradients, Jacobians, Hessians, multivariate chain rule, matrix calculus identities, Taylor series expansions. |
| [Ch06_Probability_Distributions.ipynb](Ch06_Probability_Distributions.ipynb) | Probability & Distributions | **Ex 6.1 – 6.13** (All 13) | 29 | Bayes' theorem, marginal and conditional Gaussians, Gaussian Mixture Models, multivariate change-of-variables, Probability Integral Transform (PIT), 10M-trial Monte Carlo verification. |
| [Ch07_Continuous_Optimization.ipynb](Ch07_Continuous_Optimization.ipynb) | Continuous Optimization | **Ex 7.1 – 7.11** (All 11) | 25 | Gradient descent with momentum, Stochastic Gradient Descent (SGD), Lagrange multipliers, Karush-Kuhn-Tucker (KKT) conditions, Linear Programming (LP) and Quadratic Programming (QP) duality, smoothed hinge loss. |

---

### Part II: Central Machine Learning Algorithms

| Notebook | Topic | Exercise Coverage | Cells | Key Concepts & Methods |
|:---|:---|:---:|:---:|:---|
| [Ch08_When_Models_Meet_Data.ipynb](Ch08_When_Models_Meet_Data.ipynb) | When Models Meet Data | **Ex 8.1 – 8.5** | 15 | • **Ex 8.1:** Empirical Risk Minimization (ERM) loss surfaces ($L_1, L_2$, Huber) & outlier robustness.<br>• **Ex 8.2:** MLE vs. MAP parameter estimation under Gaussian conjugate priors.<br>• **Ex 8.3:** Directed Graphical Models (DAGs), d-separation, and collider Explaining Away.<br>• **Ex 8.4:** Analytical Bias-Variance decomposition & 10,000-trial parallel Monte Carlo simulation.<br>• **Ex 8.5:** Model Selection: Parallel 5-fold CV vs. AIC / BIC information criteria. |
| [Ch09_Linear_Regression.ipynb](Ch09_Linear_Regression.ipynb) | Linear Regression | **Ex 9.1 – 9.4** | 12 | • **Ex 9.1:** Maximum Likelihood Linear Regression, Normal equations, orthogonal projector $\boldsymbol{P}$ properties (idempotency, symmetry), and residual orthogonality $\boldsymbol{\Phi}^\top \boldsymbol{e} = \boldsymbol{0}$.<br>• **Ex 9.2:** MAP Ridge regression, singular value shrinkage factor $\frac{\sigma_i^2}{\sigma_i^2+\lambda}$, and effective degrees of freedom $\operatorname{df}(\lambda)$.<br>• **Ex 9.3:** Sequential Bayesian linear regression, Gaussian conjugate posterior updating, epistemic vs. aleatoric predictive variance.<br>• **Ex 9.4:** High-degree polynomial overfitting & parallel 5-fold cross-validation. |
| [Ch10_PCA.ipynb](Ch10_PCA.ipynb) | Principal Component Analysis | **Ex 10.1 – 10.5** | 15 | • **Ex 10.1:** Dual Lagrangian derivations (Maximum Variance vs. Minimum Reconstruction Error) & PCA class from scratch.<br>• **Ex 10.2:** Data centering & standardization effects on principal axes.<br>• **Ex 10.3:** Low-rank matrix approximation & Eckart–Young–Mirsky theorem (spectral and Frobenius norms).<br>• **Ex 10.4:** High-dimensional Dual PCA ($\mathcal{O}(N^3)$ vs $\mathcal{O}(D^3)$) parallel runtime benchmark up to $D=12,000$.<br>• **Ex 10.5:** Probabilistic PCA (PPCA) & exact maximum likelihood noise variance $\sigma_{\text{ML}}^2$. |
| [Ch11_GMM.ipynb](Ch11_GMM.ipynb) | Gaussian Mixture Models | **Ex 11.1 – 11.4** | 11 | • **Ex 11.1:** Expectation-Maximization (EM) from scratch, Evidence Lower Bound (ELBO) ascent, and 2D cluster evolution.<br>• **Ex 11.2:** Hard EM and the K-Means clustering limit as variance $\sigma^2 \to 0$.<br>• **Ex 11.3:** Free parameter counting and parallel 56-restart BIC/AIC model selection ($K^*=3$).<br>• **Ex 11.4:** Covariance parameterizations: Spherical vs. Diagonal vs. Full Covariance log-likelihood hierarchy. |
| [Ch12_SVM.ipynb](Ch12_SVM.ipynb) | Support Vector Machines | **Ex 12.1 – 12.4** | 16 | • **Ex 12.1:** Hard-margin primal QP, Lagrangian dual, KKT complementary slackness, and support vector geometry.<br>• **Ex 12.2:** Soft-margin SVM, Hinge loss, and subgradient descent optimization from scratch.<br>• **Ex 12.3:** Soft-margin kernel SVM & parallel 5-fold CV grid search (100% accuracy on concentric circles).<br>• **Ex 12.4:** Mercer's theorem, SymPy symbolic proof of 6D polynomial feature map $\boldsymbol{\phi}(\boldsymbol{x})$, Gram matrix PSD verification, and infinite-dimensional RBF space proof. |

---

## 🔬 Computational Methodology & Quality Standards

Every notebook adheres strictly to the following engineering standards:
1. **Mathematical Rigor:** Complete analytical LaTeX derivations precede every code implementation.
2. **From-Scratch Implementations:** All core algorithms (Gaussian elimination, SVD, PCA, EM, Dual QP, Subgradient Descent) are built from mathematical first principles using NumPy and SciPy.
3. **Symbolic Verification:** SymPy is utilized to algebraically confirm gradient derivations, characteristic polynomials, and Mercer kernel expansions.
4. **Zero Warnings Guarantee:** All code blocks pass strict Python syntax inspection with zero `SyntaxWarning` and zero `SyntaxError` (e.g. raw strings for LaTeX escape sequences).
5. **Reproducibility:** Seeded random state (`np.random.seed(42)`) ensures deterministic execution across all environments.

---

## 🚀 Running the Notebooks

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed. The required packages include:
```bash
pip install numpy scipy matplotlib sympy psutil
```

### 2. Launching Jupyter
```bash
cd "Sem 1/Maths4AI/MML_Exercises"
jupyter notebook
```

### 3. Execution & Reproducibility
Each notebook is completely self-contained. Open and run any notebook directly in Jupyter Notebook, JupyterLab, or VS Code:
```bash
# Example: launch and run Chapter 10 (PCA) or Chapter 12 (SVM) directly
jupyter notebook Ch10_PCA.ipynb
```
