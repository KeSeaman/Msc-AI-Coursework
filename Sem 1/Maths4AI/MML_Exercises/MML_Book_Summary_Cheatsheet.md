# Mathematics for Machine Learning — Comprehensive Conceptual Summary & Mathematical Cheatsheet

<p align="center">
  <h2 align="center">Mathematics for Machine Learning</h2>
  <p align="center">
    <strong>Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong</strong><br>
    <em>Cambridge University Press (2020) &bull; Official Companion: <a href="https://mml-book.com/">mml-book.com</a></em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Textbook-Cambridge_University_Press-8A2BE2?style=for-the-badge" alt="Textbook"/>
    <img src="https://img.shields.io/badge/Coverage-Chapters_1_to_12_(Complete)-blue?style=for-the-badge" alt="Coverage"/>
    <img src="https://img.shields.io/badge/Reference-Mathematical_Cheatsheet-orange?style=for-the-badge" alt="Reference"/>
    <img src="https://img.shields.io/badge/Code_Companion-11_Jupyter_Notebooks-brightgreen?style=for-the-badge" alt="Notebooks"/>
  </p>
</p>

---

## 📌 Table of Contents

1. [Architectural Roadmap: How the Pillars Interlock](#-architectural-roadmap-how-the-pillars-interlock)
2. [Part I: Mathematical Foundations](#-part-i-mathematical-foundations)
   - [Chapter 1: Introduction and Motivation](#chapter-1-introduction-and-motivation)
   - [Chapter 2: Linear Algebra](#chapter-2-linear-algebra)
   - [Chapter 3: Analytic Geometry](#chapter-3-analytic-geometry)
   - [Chapter 4: Matrix Decompositions](#chapter-4-matrix-decompositions)
   - [Chapter 5: Vector Calculus](#chapter-5-vector-calculus)
   - [Chapter 6: Probability and Distributions](#chapter-6-probability-and-distributions)
   - [Chapter 7: Continuous Optimization](#chapter-7-continuous-optimization)
3. [Part II: Central Machine Learning Algorithms](#-part-ii-central-machine-learning-algorithms)
   - [Chapter 8: When Models Meet Data](#chapter-8-when-models-meet-data)
   - [Chapter 9: Linear Regression](#chapter-9-linear-regression)
   - [Chapter 10: Dimensionality Reduction with PCA](#chapter-10-dimensionality-reduction-with-principal-component-analysis-pca)
   - [Chapter 11: Density Estimation with GMM](#chapter-11-density-estimation-with-gaussian-mixture-models-gmm)
   - [Chapter 12: Classification with Support Vector Machines](#chapter-12-classification-with-support-vector-machines-svm)
4. [Master Reference Cards & Rapid-Fire Tables](#-master-reference-cards--rapid-fire-tables)
   - [A. Matrix Calculus Identities](#a-matrix-calculus-identities)
   - [B. Matrix Decompositions Comparison Matrix](#b-matrix-decompositions-comparison-matrix)
   - [C. Probability Distributions Quick Sheet](#c-probability-distributions-quick-sheet)
   - [D. Central ML Algorithms Comparison Matrix](#d-central-ml-algorithms-comparison-matrix)

---

## 🗺 Architectural Roadmap: How the Pillars Interlock

In *Mathematics for Machine Learning*, the discipline of machine learning is rigorously unified under a single structural equation:

$$\textbf{Machine Learning} = \textbf{Data} + \textbf{Model} + \textbf{Learning Objective (Optimization)}$$

The book is partitioned into two complementary halves:
- **Part I (Chapters 2–7):** The fundamental mathematical toolbox spanning linear algebra, analytic geometry, matrix factorizations, vector calculus, probability theory, and continuous optimization.
- **Part II (Chapters 8–12):** The four canonical machine learning problem archetypes (regression, dimensionality reduction, density estimation, and classification) formulated entirely using the tools built in Part I.

```text
+----------------------------------------------------------------------------------------------------+
|                                  PART I: MATHEMATICAL FOUNDATIONS                                  |
|                                                                                                    |
|       [Ch 2: Linear Algebra]        [Ch 3: Analytic Geometry]         [Ch 4: Decompositions]       |
|      Vectors, Spaces, Systems      Norms, Projections, Orthog.      SVD, Eigendecomp, Low-Rank     |
|                                                                                                    |
|                 |                               |                               |                  |
|                 v                               v                               v                  |
|                                                                                                    |
|      [Ch 5: Vector Calculus]       [Ch 6: Probability & Stats]         [Ch 7: Optimization]        |
|     Gradients, Jacobians, Hess     Bayes, Gaussians, Conjugacy      Lagrangians, KKT, Duality      |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                   PART II: CENTRAL ML ALGORITHMS                                   |
|                                                                                                    |
|     [Ch 8: Models & Data]     --->   Foundational Framework (ERM, MLE/MAP, Bias-Variance)          |
|     [Ch 9: Regression]        <---   Ch 2 (Linear), Ch 3 (Proj), Ch 5 (Deriv), Ch 6 (Prior)        |
|     [Ch 10: PCA]              <---   Ch 3 (Orthog), Ch 4 (SVD/Spectral), Ch 7 (Lagrangian)         |
|     [Ch 11: GMM]              <---   Ch 6 (Multivariate Gauss), Ch 7 (Jensen ELBO Ascent)          |
|     [Ch 12: SVM]              <---   Ch 3 (Margin Hyperplane), Ch 7 (Dual QP, KKT, Mercer)         |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Foundations-to-Algorithms Dependency Matrix

| Part II Algorithm | Linear Algebra (Ch 2) | Analytic Geometry (Ch 3) | Matrix Decomp. (Ch 4) | Vector Calculus (Ch 5) | Probability (Ch 6) | Continuous Optim. (Ch 7) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression (Ch 9)** | System $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}\boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{y}$ | Orthogonal Projector $\boldsymbol{P}$ | SVD shrinkage $\frac{\sigma_i^2}{\sigma_i^2 + \lambda}$ | $\nabla_{\boldsymbol{\theta}}\|\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{\theta}\|^2$ | Gaussian conjugate prior / posterior | Convex quadratic objective |
| **PCA (Ch 10)** | Basis transformation | Orthonormal basis & projection | SVD & Spectral Theorem ($\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^\top$) | Derivative of Rayleigh quotient | Probabilistic PCA (PPCA) | Lagrangian with $\|\boldsymbol{b}\|^2 = 1$ |
| **GMMs (Ch 11)** | Linear transformations | Mahalanobis metric $(\boldsymbol{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$ | Cholesky of Covariance $\boldsymbol{\Sigma} = \boldsymbol{L}\boldsymbol{L}^\top$ | Gradient of complete-data log-likelihood | Multivariate Normal, Bayes, Marginalization | Constrained optimization ($\sum \pi_k = 1$), ELBO |
| **SVMs (Ch 12)** | Affine hyperplanes | Geometric margin $\frac{2}{\|\boldsymbol{w}\|}$ | Kernel Gram matrix positive semi-definiteness | Subgradient descent of Hinge loss | Maximum margin as robust estimation | Dual Quadratic Program & KKT conditions |

---

## 🏛 Part I: Mathematical Foundations

---

### Chapter 1: Introduction and Motivation

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 1 establishes the structural philosophy of machine learning, defining it as the algorithmic synthesis of data representation, function approximation, and numerical parameter optimization. It frames the entire discipline around two complementary lenses: the *geometric perspective*, which conceptualizes inputs, parameters, and features as geometric points and subspaces embedded within high-dimensional Euclidean manifolds, and the *probabilistic perspective*, which models noise, uncertainty, and epistemic beliefs through likelihood distributions and Bayesian inference. By clarifying this duality, the chapter demonstrates how fundamental mathematical concepts like vectors, derivatives, and probabilities are not isolated abstractions, but rather the direct computational vocabulary required to formulate, fit, and evaluate modern learning algorithms.

#### Core Concepts & Intuition
- **Data as Vectors:** An object (image, text embedding, audio sample, sensor readout) is represented as a point $\boldsymbol{x} \in \mathbb{R}^D$ in a high-dimensional vector space.
- **Models as Functions:** A hypothesis $f(\boldsymbol{x}; \boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$ maps inputs to predictions or probability densities.
- **Learning as Parameter Search:** Finding the parameter vector $\boldsymbol{\theta}^\star$ that minimizes an empirical loss function or maximizes a posterior likelihood.
- **The Two Core Perspectives:**
  1. **Geometric View:** Focuses on distances, projections, angles, linear subspaces, and manifold structures in $\mathbb{R}^D$.
  2. **Probabilistic View:** Treats parameters and observations as random variables, modeling uncertainty, noise, likelihoods, and epistemic beliefs.

#### Standard Dimension Notations
- $N$: Number of observations (sample size / data points).
- $D$: Dimensionality of input space $\mathbb{R}^D$.
- $M$: Dimensionality of target feature subspace or latent space ($M < D$).
- $K$: Number of discrete classes, mixture components, or clusters.

---

### Chapter 2: Linear Algebra

> **Companion Notebook:** [`Ch02_Linear_Algebra.ipynb`](Ch02_Linear_Algebra.ipynb) &bull; 50 cells &bull; Exercises 2.1 to 2.20

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 2 provides the algebraic backbone of machine learning by formalizing how high-dimensional observations and transformations are mathematically manipulated as vectors, matrices, and linear systems. Beginning with Gaussian elimination and the classification of linear systems ($\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$), the chapter systematically develops abstract vector spaces, linear combinations, bases, and coordinate transformations, establishing how high-dimensional features can be re-expressed across alternative coordinate frames. It culminates in the study of linear mappings, defining the kernel (null space) and image (column space) of transformation matrices, and proves the foundational Rank-Nullity Theorem, which governs the degrees of freedom preserved or collapsed when data undergoes linear projections in neural layers and dimensionality reduction pipelines.

#### 1. Systems of Linear Equations
A system of $M$ linear equations in $N$ unknowns is compactly written in matrix notation:

$$\boldsymbol{A}\boldsymbol{x} = \boldsymbol{b}, \quad \boldsymbol{A} \in \mathbb{R}^{M \times N}, \; \boldsymbol{x} \in \mathbb{R}^N, \; \boldsymbol{b} \in \mathbb{R}^M$$

- **Augmented Matrix:** $[\boldsymbol{A} \mid \boldsymbol{b}] \in \mathbb{R}^{M \times (N+1)}$.
- **Elementary Row Operations:** (1) Row swap, (2) Row multiplication by non-zero scalar, (3) Adding a scalar multiple of one row to another.
- **Row-Echelon Form (REF):** Pivots move strictly to the right; all entries beneath a pivot are zero.
- **Reduced Row-Echelon Form (RREF):** Every pivot is 1, and each pivot is the sole non-zero entry in its column.
- **Solution Regimes:**
  - $\text{rank}(\boldsymbol{A}) < \text{rank}([\boldsymbol{A} \mid \boldsymbol{b}])$: **No solution** (inconsistent).
  - $\text{rank}(\boldsymbol{A}) = \text{rank}([\boldsymbol{A} \mid \boldsymbol{b}]) = N$: **Unique solution**.
  - $\text{rank}(\boldsymbol{A}) = \text{rank}([\boldsymbol{A} \mid \boldsymbol{b}]) < N$: **Infinitely many solutions** with $N - \text{rank}(\boldsymbol{A})$ free parameters.

#### 2. Vector Spaces, Linear Independence & Basis
- **Vector Space Axioms:** Closed under vector addition $(\boldsymbol{x} + \boldsymbol{y} \in V)$ and scalar multiplication $(\lambda \boldsymbol{x} \in V)$, satisfying associativity, commutativity, additive identity ($\boldsymbol{0}$), additive inverse ($-\boldsymbol{x}$), and distributivity.
- **Linear Independence:** The vectors $\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_k\}$ are linearly independent if:

$$\sum_{i=1}^k \lambda_i \boldsymbol{x}_i = \boldsymbol{0} \iff \lambda_1 = \lambda_2 = \dots = \lambda_k = 0$$

- **Basis:** A minimal generating set of $V$. A set $\mathcal{B} = \{\boldsymbol{b}_1, \dots, \boldsymbol{b}_D\}$ is a basis of $V$ iff (1) $\mathcal{B}$ is linearly independent, and (2) $\text{span}(\mathcal{B}) = V$.
- **Dimension:** The unique number of basis vectors spanning $V$, denoted $\dim(V)$.

#### 3. Linear Mappings & Fundamental Spaces
A mapping $\Phi: V \to W$ is linear if $\Phi(\lambda \boldsymbol{x} + \mu \boldsymbol{y}) = \lambda \Phi(\boldsymbol{x}) + \mu \Phi(\boldsymbol{y})$.
- **Transformation Matrix:** Given ordered bases $B = (\boldsymbol{b_1}, \dots, \boldsymbol{b_N})$ of $V$ and $C = (\boldsymbol{c_1}, \dots, \boldsymbol{c_M})$ of $W$, $\Phi$ has a unique representation matrix $\boldsymbol{A}_\Phi \in \mathbb{R}^{M \times N}$.
- **Change of Basis:** If $\boldsymbol{S}$ transforms basis $B \to \tilde{B}$ in $V$ and $\boldsymbol{T}$ transforms $C \to \tilde{C}$ in $W$:

$$\tilde{\boldsymbol{A}}_\Phi = \boldsymbol{T}^{-1}\boldsymbol{A}_\Phi \boldsymbol{S}$$

- **Kernel (Null Space):** $\ker(\Phi) = \{\boldsymbol{x} \in V \mid \Phi(\boldsymbol{x}) = \boldsymbol{0}\}$.
- **Image (Range / Column Space):** $\text{im}(\Phi) = \{\boldsymbol{y} \in W \mid \exists \boldsymbol{x} \in V: \Phi(\boldsymbol{x}) = \boldsymbol{y}\}$.
- **Rank-Nullity Theorem (Fundamental Theorem of Linear Mappings):**

$$\dim(V) = \dim(\ker(\Phi)) + \dim(\text{im}(\Phi)) = \text{nullity}(\boldsymbol{A}) + \text{rank}(\boldsymbol{A})$$

#### 4. Affine Spaces
- An affine space $L = \boldsymbol{x}_0 + U$ is a vector subspace $U \subseteq V$ shifted by a support vector $\boldsymbol{x}_0$.
- In general, affine spaces do not pass through the origin and are not closed under standard vector addition.

---

### Chapter 3: Analytic Geometry

> **Companion Notebook:** [`Ch03_Analytic_Geometry.ipynb`](Ch03_Analytic_Geometry.ipynb) &bull; 27 cells &bull; Exercises 3.1 to 3.10

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 3 equips linear algebra with metric geometry, introducing rigorous mathematical machinery to measure vector lengths, angular separations, distances, and orthogonal relationships in high-dimensional vector spaces. By defining inner products as symmetric positive-definite bilinear forms and deriving induced $L_p$ norms, the chapter proves the pivotal Cauchy-Schwarz inequality, providing the theoretical bedrock for cosine similarity, kernel metrics, and geometric margin maximization. Most critically for machine learning, it formalizes orthogonal projections onto arbitrary linear subspaces via projection matrices $\boldsymbol{P}_\pi = \boldsymbol{B}(\boldsymbol{B}^\top \boldsymbol{B})^{-1}\boldsymbol{B}^\top$, providing the exact geometric mechanism that underlies ordinary least-squares regression, principal component analysis, and linear subspace approximations.

#### 1. Norms & Distances
A norm $\|\cdot\|: V \to \mathbb{R}$ quantifies vector length and satisfies:
1. Positive definiteness: $\|\boldsymbol{x}\| \ge 0$, and $\|\boldsymbol{x}\| = 0 \iff \boldsymbol{x} = \boldsymbol{0}$.
2. Absolute homogeneity: $\|\lambda \boldsymbol{x}\| = |\lambda| \|\boldsymbol{x}\|$.
3. Triangle inequality: $\|\boldsymbol{x} + \boldsymbol{y}\| \le \|\boldsymbol{x}\| + \|\boldsymbol{y}\|$.

- **$L_p$-Norm:**

$$\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^D |x_i|^p\right)^{1/p}$$

  - $L_1$-Norm (Manhattan): $\|\boldsymbol{x}\|_1 = \sum |x_i|$ (promotes sparsity, Lasso regularization).
  - $L_2$-Norm (Euclidean): $\|\boldsymbol{x}\|_2 = \sqrt{\sum x_i^2} = \sqrt{\boldsymbol{x}^\top \boldsymbol{x}}$ (smooth, rotationally invariant, Ridge regularization).
  - $L_\infty$-Norm (Chebyshev / Max): $\|\boldsymbol{x}\|_\infty = \max_i |x_i|$.

#### 2. Inner Products & Angles
An inner product $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ is a symmetric, positive-definite bilinear form.
- Every inner product on $\mathbb{R}^D$ can be written as:

$$\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{y}, \quad \boldsymbol{A} \in \mathbb{R}^{D \times D} \text{ symmetric positive-definite } (\boldsymbol{A} \succ 0)$$

- **Cauchy-Schwarz Inequality:**

$$|\langle \boldsymbol{x}, \boldsymbol{y} \rangle| \le \|\boldsymbol{x}\| \|\boldsymbol{y}\|$$

- **Angle $\omega$ between vectors:**

$$\cos \omega = \frac{\langle \boldsymbol{x}, \boldsymbol{y} \rangle}{\|\boldsymbol{x}\| \|\boldsymbol{y}\|}$$

- **Orthogonality:** $\boldsymbol{x} \perp \boldsymbol{y} \iff \langle \boldsymbol{x}, \boldsymbol{y} \rangle = 0 \iff \cos \omega = 0$.

#### 3. Orthogonal Projections
Projecting an arbitrary vector $\boldsymbol{x} \in \mathbb{R}^D$ onto a subspace $U = \text{span}(\boldsymbol{B})$ where columns of $\boldsymbol{B} = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_M] \in \mathbb{R}^{D \times M}$ form a basis:
- **Projection Coordinates $\boldsymbol{\lambda}^\star \in \mathbb{R}^{M}$:**

$$\boldsymbol{\lambda}^\star = (\boldsymbol{B}^\top \boldsymbol{B})^{-1}\boldsymbol{B}^\top \boldsymbol{x}$$

- **Projected Vector $\pi_U(\boldsymbol{x}) \in \mathbb{R}^D$:**

$$\pi_U(\boldsymbol{x}) = \boldsymbol{B}\boldsymbol{\lambda}^\star = \boldsymbol{B}(\boldsymbol{B}^\top \boldsymbol{B})^{-1}\boldsymbol{B}^\top \boldsymbol{x}$$

- **Orthogonal Projection Matrix $\boldsymbol{P}_\pi$:**

$$\boldsymbol{P}_\pi = \boldsymbol{B}(\boldsymbol{B}^\top \boldsymbol{B})^{-1}\boldsymbol{B}^\top$$

> [!TIP]
> **Properties of Projection Matrices ($\boldsymbol{P} = \boldsymbol{P}_\pi$):**
> 1. **Idempotence:** $\boldsymbol{P}^2 = \boldsymbol{P}$ (projecting twice does not alter the projection).
> 2. **Symmetry:** $\boldsymbol{P}^\top = \boldsymbol{P}$ (for orthogonal projections).
> 3. **Orthogonality of Error:** The residual vector $\boldsymbol{e} = \boldsymbol{x} - \boldsymbol{P}\boldsymbol{x}$ satisfies $\boldsymbol{B}^\top \boldsymbol{e} = \boldsymbol{0}$.

#### 4. Gram-Schmidt Orthonormalization
Given a basis $(\boldsymbol{v}_1, \dots, \boldsymbol{v}_M)$, construct an orthonormal basis $(\boldsymbol{u}_1, \dots, \boldsymbol{u}_M)$:

$$\boldsymbol{u}_1 = \frac{\boldsymbol{v}_1}{\|\boldsymbol{v}_1\|}, \quad \tilde{\boldsymbol{u}}_k = \boldsymbol{v}_k - \sum_{j=1}^{k-1} \langle \boldsymbol{v}_k, \boldsymbol{u}_j \rangle \boldsymbol{u}_j, \quad \boldsymbol{u}_k = \frac{\tilde{\boldsymbol{u}}_k}{\|\tilde{\boldsymbol{u}}_k\|}$$

#### 5. Rotations in $\mathrm{SO}(n)$
- Orthogonal matrix $\boldsymbol{R}$: $\boldsymbol{R}^\top \boldsymbol{R} = \boldsymbol{I}$, preserving lengths and angles: $\|\boldsymbol{R}\boldsymbol{x}\| = \|\boldsymbol{x}\|$.
- Proper rotations: $\det(\boldsymbol{R}) = +1$ (Group $\mathrm{SO}(n)$).
- **2D Counterclockwise Rotation by $\theta$:**

$$\boldsymbol{R}(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$$

---

### Chapter 4: Matrix Decompositions

> **Companion Notebook:** [`Ch04_Matrix_Decompositions.ipynb`](Ch04_Matrix_Decompositions.ipynb) &bull; 29 cells &bull; Exercises 4.1 to 4.12

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 4 investigates matrix factorizations, revealing how complex, entangled linear operators can be decomposed into canonical, interpretable geometric operations such as rotations, coordinate scalings, and triangular systems. Covering determinants as volume scaling factors and traces as cyclic invariants, the chapter details the characteristic polynomial, eigenvalue problems, and the Spectral Theorem for symmetric matrices ($\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^\top$). It then develops the Cholesky decomposition ($\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}^\top$) for symmetric positive-definite matrices—essential for sampling multivariate Gaussians and solving normal equations—and culminates in the Singular Value Decomposition (SVD) and the Eckart–Young–Mirsky Theorem, establishing the mathematically optimal rank-$k$ low-rank approximation framework for data compression, pseudoinverses, and latent factor discovery.

#### 1. Determinant & Trace
- **Determinant:** Geometric volume scaling factor of the linear mapping.
  - $\det(\boldsymbol{A}\boldsymbol{B}) = \det(\boldsymbol{A})\det(\boldsymbol{B})$.
  - $\det(\boldsymbol{A}^{-1}) = \frac{1}{\det(\boldsymbol{A})}$.
  - $\det(\boldsymbol{A}) = \prod_{i=1}^D \lambda_i$ (product of eigenvalues).
  - $\det(\boldsymbol{A}) \ne 0 \iff \boldsymbol{A}$ is invertible.
- **Trace:** Sum of the diagonal elements $\mathrm{tr}(\boldsymbol{A}) = \sum_{i=1}^D a_{ii}$.
  - Cyclic permutation property: $\mathrm{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}) = \mathrm{tr}(\boldsymbol{B}\boldsymbol{C}\boldsymbol{A}) = \mathrm{tr}(\boldsymbol{C}\boldsymbol{A}\boldsymbol{B})$.
  - $\mathrm{tr}(\boldsymbol{A}) = \sum_{i=1}^D \lambda_i$ (sum of eigenvalues).

#### 2. Eigenvalues, Eigenvectors & Diagonalization
- **Eigenvalue Equation:** $\boldsymbol{A}\boldsymbol{x} = \lambda \boldsymbol{x}, \; \boldsymbol{x} \ne \boldsymbol{0}$.
- **Characteristic Polynomial:** $p_{\boldsymbol{A}}(\lambda) = \det(\boldsymbol{A} - \lambda \boldsymbol{I}) = 0$.
- **Diagonalization:** An $N \times N$ matrix $\boldsymbol{A}$ is diagonalizable iff it has $N$ linearly independent eigenvectors:

$$\boldsymbol{A} = \boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{-1}$$

where $\boldsymbol{D} = \text{diag}(\lambda_1, \dots, \lambda_N)$ and $\boldsymbol{P} = [\boldsymbol{p}_1 \mid \dots \mid \boldsymbol{p}_N]$.
- **Spectral Theorem for Symmetric Matrices:** Every real symmetric matrix $\boldsymbol{A} = \boldsymbol{A}^\top$ can be orthogonally diagonalized:

$$\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^\top = \sum_{i=1}^D \lambda_i \boldsymbol{q}_i \boldsymbol{q}_i^\top, \quad \boldsymbol{Q}^\top \boldsymbol{Q} = \boldsymbol{I}, \; \lambda_i \in \mathbb{R}$$

#### 3. Cholesky Decomposition
For any symmetric positive-definite matrix $\boldsymbol{A} \succ 0$:

$$\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}^\top$$

where $\boldsymbol{L}$ is a unique lower triangular matrix with strictly positive diagonal entries.
- Used extensively in Gaussian sampling: If $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$, then $\boldsymbol{x} = \boldsymbol{\mu} + \boldsymbol{L}\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{L}\boldsymbol{L}^\top = \boldsymbol{\Sigma})$.

#### 4. Singular Value Decomposition (SVD)
For **any** real matrix $\boldsymbol{A} \in \mathbb{R}^{M \times N}$:

$$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^\top$$

- $\boldsymbol{U} \in \mathbb{R}^{M \times M}$: Orthonormal eigenvectors of $\boldsymbol{A}\boldsymbol{A}^\top$ (left-singular vectors).
- $\boldsymbol{\Sigma} \in \mathbb{R}^{M \times N}$: Diagonal matrix of singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, where $\sigma_i = \sqrt{\lambda_i(\boldsymbol{A}^\top \boldsymbol{A})}$.
- $\boldsymbol{V} \in \mathbb{R}^{N \times N}$: Orthonormal eigenvectors of $\boldsymbol{A}^\top \boldsymbol{A}$ (right-singular vectors).
- **Geometric Interpretation:** A sequence of three transformations: (1) Rotation in $\mathbb{R}^N$ by $\boldsymbol{V}^\top$, (2) Scaling along coordinate axes by $\sigma_i$, (3) Rotation in $\mathbb{R}^M$ by $\boldsymbol{U}$.

#### 5. Eckart-Young-Mirsky Theorem (Low-Rank Approximation)
The optimal rank-$k$ approximation ($k < r$) to $\boldsymbol{A}$ under both the Spectral norm $\|\cdot\|_2$ and the Frobenius norm $\|\cdot\|_F$ is the truncated SVD:

$$\boldsymbol{A}_k = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^\top = \arg\min_{\text{rank}(\boldsymbol{B}) \le k} \|\boldsymbol{A} - \boldsymbol{B}\|$$

- **Reconstruction Error:**

$$\|\boldsymbol{A} - \boldsymbol{A}_k\|_2 = \sigma_{k+1}, \qquad \|\boldsymbol{A} - \boldsymbol{A}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

#### 6. Moore-Penrose Pseudoinverse
For an arbitrary matrix $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top$:

$$\boldsymbol{A}^\dagger = \boldsymbol{V}\boldsymbol{\Sigma}^\dagger \boldsymbol{U}^\top$$

where $\boldsymbol{\Sigma}^\dagger$ is formed by replacing all non-zero singular values with $1/\sigma_i$ and transposing.
- If $\boldsymbol{A}$ has full column rank: $\boldsymbol{A}^\dagger = (\boldsymbol{A}^\top \boldsymbol{A})^{-1}\boldsymbol{A}^\top$ (Left inverse).
- If $\boldsymbol{A}$ has full row rank: $\boldsymbol{A}^\dagger = \boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{A}^\top)^{-1}$ (Right inverse).

---

### Chapter 5: Vector Calculus

> **Companion Notebook:** [`Ch05_Vector_Calculus.ipynb`](Ch05_Vector_Calculus.ipynb) &bull; 22 cells &bull; Exercises 5.1 to 5.9

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 5 develops differential calculus on multi-dimensional manifolds and matrix spaces, constructing the core analytical machinery that enables gradient-based optimization throughout machine learning. It generalizes univariate differentiation to gradients of scalar fields, Jacobian matrices of vector-valued mappings, and Hessian matrices that capture curvature and local convexity. Crucially, the chapter formalizes the multivariate chain rule and reverse-mode automatic differentiation (backpropagation), proving why computing vector-Jacobian products from output to input scales linearly $(\mathcal{O}(P))$ rather than quadratically with network parameter counts. It concludes with essential matrix calculus identities for traces, determinants, and quadratic forms, providing the exact derivation tools needed to compute analytical gradients for maximum likelihood and regularized loss objectives.

#### 1. Gradients, Jacobians & Hessians
- **Gradient (Scalar field $f: \mathbb{R}^D \to \mathbb{R}$):** Column vector of partial derivatives:

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_D} \end{bmatrix} \in \mathbb{R}^D$$

> Pointing in the direction of maximal ascent; orthogonal to level curves $f(\boldsymbol{x}) = c$.

- **Jacobian Matrix (Vector field $\boldsymbol{f}: \mathbb{R}^D \to \mathbb{R}^E$):**

$$\boldsymbol{J} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_E}{\partial x_1} & \dots & \frac{\partial f_E}{\partial x_D} \end{bmatrix} \in \mathbb{R}^{E \times D}$$

- **Hessian Matrix (Curvature of scalar field $f: \mathbb{R}^D \to \mathbb{R}$):**

$$\boldsymbol{H} = \nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x}) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_D \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_D^2} \end{bmatrix} \in \mathbb{R}^{D \times D}$$

- **Schwarz's Theorem:** If partials are continuous, $\boldsymbol{H} = \boldsymbol{H}^\top$ (symmetric).
- **Convexity Test:** $f$ is strictly convex on an open convex set iff $\boldsymbol{H}(\boldsymbol{x}) \succ 0$ everywhere.

#### 2. Multivariate Chain Rule & Automatic Differentiation
For composite function $\boldsymbol{f}(\boldsymbol{g}(\boldsymbol{x}))$ where $\boldsymbol{g}: \mathbb{R}^D \to \mathbb{R}^K$ and $\boldsymbol{f}: \mathbb{R}^K \to \mathbb{R}^E$:

$$\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{g}} \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}} \quad \iff \quad \boldsymbol{J}_{\boldsymbol{f} \circ \boldsymbol{g}} = \boldsymbol{J}_{\boldsymbol{f}} \boldsymbol{J}_{\boldsymbol{g}}$$

- **Reverse-Mode AD (Backpropagation):** Computes Vector-Jacobian Products (VJPs) from output to input. For neural networks with scalar loss $L \in \mathbb{R}$ and millions of parameters $\boldsymbol{\theta} \in \mathbb{R}^P$, reverse-mode requires **1 backward sweep** $(\mathcal{O}(P))$, whereas forward-mode would require $P$ passes $(\mathcal{O}(P^2))$.

#### 3. Multivariate Taylor Series
Taylor expansion of $f: \mathbb{R}^D \to \mathbb{R}$ around $\boldsymbol{x}_0$:

$$f(\boldsymbol{x}) \approx f(\boldsymbol{x}_0) + \nabla f(\boldsymbol{x}_0)^\top (\boldsymbol{x} - \boldsymbol{x}_0) + \frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}_0)^\top \boldsymbol{H}(\boldsymbol{x}_0)(\boldsymbol{x} - \boldsymbol{x}_0) + \dots$$

---

### Chapter 6: Probability and Distributions

> **Companion Notebook:** [`Ch06_Probability_Distributions.ipynb`](Ch06_Probability_Distributions.ipynb) &bull; 29 cells &bull; Exercises 6.1 to 6.13

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 6 provides the axiomatic foundations of probability theory and statistical modeling, offering a principled mathematical framework for quantifying aleatoric data noise and epistemic model uncertainty. Structured around Kolmogorov's axioms, the sum rule (marginalization), the product rule, and Bayes' theorem, the chapter thoroughly characterizes the Multivariate Gaussian distribution, providing complete block-matrix partition derivations for marginalization and conditioning via Schur complements. It explores the exponential family, conjugate priors, and conjugate updating schemes (such as Beta-Binomial and Gaussian-Gaussian), and formalizes the multi-dimensional change-of-variables theorem with Jacobian determinants, establishing the theoretical mechanics underpinning normalization flows, probability integral transforms, and generative probabilistic modeling.

#### 1. Kolmogorov Axioms & Fundamental Rules
- **Axioms of Probability:** (1) $P(A) \ge 0$, (2) $P(\Omega) = 1$, (3) Countable additivity for disjoint events: $P(\bigcup_i A_i) = \sum_i P(A_i)$.
- **Sum Rule (Marginalization):**

$$p(x) = \int p(x, y)\,dy \quad (\text{continuous}), \quad p(x) = \sum_y p(x, y) \quad (\text{discrete})$$

- **Product Rule:**

$$p(x, y) = p(x|y)p(y) = p(y|x)p(x)$$

- **Bayes' Theorem:**

$$p(\boldsymbol{\theta}|\mathcal{D}) = \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})} = \frac{p(\mathcal{D}|\boldsymbol{\theta})p(\boldsymbol{\theta})}{\int p(\mathcal{D}|\boldsymbol{\theta}')p(\boldsymbol{\theta}')\,d\boldsymbol{\theta}'}$$

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence (Marginal Likelihood)}}$$

#### 2. Multivariate Gaussian Distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
The probability density function for $\boldsymbol{x} \in \mathbb{R}^D$:

$$p(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\right)$$

- **Mahalanobis Distance:** $\Delta = \sqrt{(\boldsymbol{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{x} - \boldsymbol{\mu})}$.
- **Linear Transformation:** If $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:

$$\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{b} \implies \boldsymbol{y} \sim \mathcal{N}(\boldsymbol{A}\boldsymbol{\mu} + \boldsymbol{b}, \boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^\top)$$

- **Marginalization and Conditioning of Joint Gaussians:**
Let the partitioned Gaussian vector be:

$$\boldsymbol{x} = \begin{bmatrix}\boldsymbol{x}_a \\ \boldsymbol{x}_b\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}\boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b\end{bmatrix}, \begin{bmatrix}\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}\right)$$

  - **Marginal Distribution:**

$$p(\boldsymbol{x}_a) = \mathcal{N}(\boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})$$

  - **Conditional Distribution:**

$$p(\boldsymbol{x}_a \mid \boldsymbol{x}_b) = \mathcal{N}(\boldsymbol{\mu}_{a|b}, \boldsymbol{\Sigma}_{a|b})$$

$$\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}(\boldsymbol{x}_b - \boldsymbol{\mu}_b)$$

$$\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$$

> [!NOTE]
> The term $\boldsymbol{\Sigma_{a|b}}$ is the **Schur complement** of $\boldsymbol{\Sigma_{bb}}$ in $\boldsymbol{\Sigma}$. Notice that $\boldsymbol{\Sigma_{a|b}}$ is completely independent of the observed value $\boldsymbol{x_b}$!

#### 3. Conjugacy & Exponential Families
- **Conjugate Prior:** A prior $p(\boldsymbol{\theta})$ is conjugate to the likelihood $p(\mathcal{D}|\boldsymbol{\theta})$ if the posterior $p(\boldsymbol{\theta}|\mathcal{D})$ belongs to the same family of distributions as the prior.
- **Canonical Exponential Family:**

$$p(\boldsymbol{x} \mid \boldsymbol{\theta}) = h(\boldsymbol{x})\exp\left(\boldsymbol{\eta}(\boldsymbol{\theta})^\top \boldsymbol{T}(\boldsymbol{x}) - A(\boldsymbol{\theta})\right)$$

where $\boldsymbol{\eta}(\boldsymbol{\theta})$ is natural parameter vector, $\boldsymbol{T}(\boldsymbol{x})$ is sufficient statistic, $h(\boldsymbol{x})$ is base measure, and $A(\boldsymbol{\theta})$ is log-partition function.

#### 4. Change of Variables / Probability Integral Transform
For a continuous random variable $\boldsymbol{X}$ and invertible transformation $\boldsymbol{Y} = \boldsymbol{g}(\boldsymbol{X})$ with inverse $\boldsymbol{X} = \boldsymbol{g}^{-1}(\boldsymbol{Y})$:

$$p_{\boldsymbol{Y}}(\boldsymbol{y}) = p_{\boldsymbol{X}}(\boldsymbol{g}^{-1}(\boldsymbol{y})) \cdot \left|\det\left(\frac{\partial \boldsymbol{g}^{-1}}{\partial \boldsymbol{y}}\right)\right| = p_{\boldsymbol{X}}(\boldsymbol{x}) \cdot \left|\det\left(\boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{x})\right)\right|^{-1}$$

- **Probability Integral Transform (PIT):** If $X$ has continuous CDF $F_X(x)$, then $U = F_X(X) \sim \text{Uniform}(0, 1)$. Conversely, $F_X^{-1}(U) \sim X$ enables arbitrary univariate random sampling from uniform draws.

---

### Chapter 7: Continuous Optimization

> **Companion Notebook:** [`Ch07_Continuous_Optimization.ipynb`](Ch07_Continuous_Optimization.ipynb) &bull; 25 cells &bull; Exercises 7.1 to 7.11

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 7 addresses continuous numerical optimization, bridging mathematical formulations to computable algorithms by exploring how to locate optimal parameter vectors in unconstrained and constrained objective landscapes. Beginning with first-order and second-order stationarity conditions, it analyzes gradient descent, heavy-ball momentum, and stochastic gradient descent (SGD), explaining how mini-batch gradient estimators navigate high-dimensional non-convex loss surfaces. The chapter then tackles constrained optimization via the method of Lagrange multipliers, proves the four Karush–Kuhn–Tucker (KKT) conditions governing inequality constraints and complementary slackness, and develops Lagrangian duality theory (weak and strong duality under Slater's condition), establishing the exact mathematical scaffolding required for quadratic programming, linear programming, and dual Support Vector Machines.

#### 1. Unconstrained Optimization & Gradient Descent
- **First-Order Necessary Condition:** $\nabla f(\boldsymbol{x}^\star) = \boldsymbol{0}$ (stationary point).
- **Second-Order Sufficient Condition:** $\nabla^2 f(\boldsymbol{x}^\star) \succ 0$ (strict local minimum).
- **Gradient Descent Updates:**
  - Standard GD: $\boldsymbol{x_{t+1}} = \boldsymbol{x_t} - \gamma \nabla f(\boldsymbol{x_t})$.
  - Heavy-Ball Momentum: $\boldsymbol{v_{t+1}} = \beta \boldsymbol{v_t} + \gamma \nabla f(\boldsymbol{x_t})$, $\boldsymbol{x_{t+1}} = \boldsymbol{x_t} - \boldsymbol{v_{t+1}}$.
  - Stochastic Gradient Descent (SGD): Uses single-sample or mini-batch unbiased gradient estimate $\mathbb{E}[\nabla f_i(\boldsymbol{x})] = \nabla f(\boldsymbol{x})$.

#### 2. Constrained Optimization & Karush-Kuhn-Tucker (KKT) Conditions
Consider the general non-linear programming problem:

$$\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{subject to} \quad g_i(\boldsymbol{x}) \le 0 \; (i=1,\dots,m), \quad h_j(\boldsymbol{x}) = 0 \; (j=1,\dots,p)$$

- **Lagrangian Function:**

$$\mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\boldsymbol{x}) + \sum_{i=1}^m \lambda_i g_i(\boldsymbol{x}) + \sum_{j=1}^p \nu_j h_j(\boldsymbol{x})$$

where $\lambda_i \ge 0$ are the KKT multipliers for inequality constraints and $\nu_j \in \mathbb{R}$ are Lagrange multipliers for equality constraints.

#### The 4 KKT Conditions (Necessary for local optimality; sufficient under convexity):
1. **Stationarity:**

$$\nabla_{\boldsymbol{x}}\mathcal{L}(\boldsymbol{x}^\star, \boldsymbol{\lambda}^\star, \boldsymbol{\nu}^\star) = \nabla f(\boldsymbol{x}^\star) + \sum_{i=1}^m \lambda_i^\star \nabla g_i(\boldsymbol{x}^\star) + \sum_{j=1}^p \nu_j^\star \nabla h_j(\boldsymbol{x}^\star) = \boldsymbol{0}$$

2. **Primal Feasibility:**

$$g_i(\boldsymbol{x}^\star) \le 0 \quad (\forall i=1,\dots,m), \quad h_j(\boldsymbol{x}^\star) = 0 \quad (\forall j=1,\dots,p)$$

3. **Dual Feasibility:**

$$\lambda_i^\star \ge 0 \quad (\forall i=1,\dots,m)$$

4. **Complementary Slackness:**

$$\lambda_i^\star g_i(\boldsymbol{x}^\star) = 0 \quad (\forall i=1,\dots,m)$$

> [!IMPORTANT]
> **Complementary Slackness Meaning:**  
> For each inequality constraint $i$, either:
> - $\lambda_i^\star = 0$: The constraint is **inactive** ($g_i(\boldsymbol{x}^\star) < 0$). Removing the constraint has no local effect.
> - $g_i(\boldsymbol{x}^\star) = 0$: The constraint is **active** (lies strictly on boundary), and $\lambda_i^\star \ge 0$.  
> This directly underpins the concept of **Support Vectors** in Chapter 12!

#### 3. Duality Theory
- **Lagrange Dual Function:** $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\boldsymbol{x}} \mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ is always concave, even if $f$ is non-convex.
- **Dual Problem:** $\max_{\boldsymbol{\lambda} \ge \boldsymbol{0}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu})$.
- **Weak Duality:** $d^\star \le p^\star$ (Dual optimal is always a lower bound on primal optimal).
- **Strong Duality ($d^\star = p^\star$):** Zero duality gap. Holds when the primal problem is convex and **Slater's condition** is satisfied (there exists a strictly feasible point $\boldsymbol{x}$ such that $g_i(\boldsymbol{x}) < 0$ for all non-affine inequalities).

---

## 🤖 Part II: Central Machine Learning Algorithms

---

### Chapter 8: When Models Meet Data

> **Companion Notebook:** [`Ch08_When_Models_Meet_Data.ipynb`](Ch08_When_Models_Meet_Data.ipynb) &bull; 15 cells &bull; Exercises 8.1 to 8.5

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 8 serves as the conceptual gateway to Part II, formally integrating the mathematical foundations of Part I to establish how theoretical model classes interface with finite, noisy empirical datasets. It formalizes Empirical Risk Minimization (ERM) under various loss functions ($L_1, L_2$, Huber), contextualizing Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) as principled statistical approaches to parameter estimation under conjugate priors. Using Directed Graphical Models (DAGs), it demystifies conditional independence, d-separation, and collider "Explaining Away" phenomena. Finally, the chapter presents the analytical Bias-Variance decomposition, rigorously explaining how model capacity dictates the trade-off between underfitting and overfitting, and details model selection criteria including $K$-fold cross-validation, AIC, and BIC.

#### 1. Empirical Risk Minimization (ERM)
- **Population Risk:** $R(f) = \mathbb{E}_{(\boldsymbol{x}, y) \sim p(\boldsymbol{x}, y)}[L(y, f(\boldsymbol{x}))]$.
- **Empirical Risk:** $R_{\text{emp}}(f) = \frac{1}{N}\sum_{n=1}^N L(y_n, f(\boldsymbol{x}_n))$.
- **Loss Functions:**
  - Squared Loss ($L_2$): $L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$ (leads to mean estimation; sensitive to outliers).
  - Absolute Loss ($L_1$): $L(y, \hat{y}) = |y - \hat{y}|$ (leads to median estimation; robust to outliers).
  - Huber Loss: Quadratic for $|y - \hat{y}| \le \delta$, linear for $|y - \hat{y}| > \delta$ (combines smoothness and robustness).

#### 2. Parameter Estimation: MLE vs. MAP
- **Maximum Likelihood Estimation (MLE):**

$$\boldsymbol{\theta}_{\text{ML}} = \arg\max_{\boldsymbol{\theta}} \sum_{n=1}^N \log p(\boldsymbol{x}_n \mid \boldsymbol{\theta})$$

- **Maximum A Posteriori (MAP):**

$$\boldsymbol{\theta}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[\sum_{n=1}^N \log p(\boldsymbol{x}_n \mid \boldsymbol{\theta}) + \log p(\boldsymbol{\theta})\right]$$

> A zero-mean Gaussian prior $p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{0}, \alpha^2 \boldsymbol{I})$ induces an $L_2$ weight penalty (Ridge): $-\frac{1}{2\alpha^2}\|\boldsymbol{\theta}\|^2$.  
> A zero-mean Laplace prior $p(\boldsymbol{\theta}) \propto \exp(-\lambda \|\boldsymbol{\theta}\|_1)$ induces an $L_1$ penalty (Lasso).

#### 3. Directed Graphical Models & D-Separation
Joint distribution factors according to Directed Acyclic Graph (DAG):

$$p(x_1, \dots, x_K) = \prod_{k=1}^K p(x_k \mid \text{parents}(x_k))$$

Three basic node motifs:
1. **Head-to-Tail (Chain):** $A \to C \to B$. Observing $C$ blocks the path: $A \perp B \mid C$.
2. **Tail-to-Tail (Common Cause):** $A \leftarrow C \to B$. Observing $C$ blocks the path: $A \perp B \mid C$.
3. **Head-to-Head (Collider / V-Structure):** $A \to C \leftarrow B$. Unconditioned, path is blocked: $A \perp B$. Observing $C$ (or any descendant of $C$) **activates** the path: $A \not\perp B \mid C$. Known as **Explaining Away**.

#### 4. Analytical Bias-Variance Decomposition
For regression with target $y = f(\boldsymbol{x}) + \epsilon$ ($\mathbb{E}[\epsilon]=0, \mathbb{V}[\epsilon]=\sigma^2$) and estimator $\hat{f}(\boldsymbol{x})$ trained on dataset $\mathcal{D}$:

$$\mathbb{E}_{\mathcal{D}, \epsilon}\left[(y - \hat{f}(\boldsymbol{x}))^2\right] = \underbrace{\left(\mathbb{E}_{\mathcal{D}}[\hat{f}(\boldsymbol{x})] - f(\boldsymbol{x})\right)^2}_{\textbf{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}\left[\left(\hat{f}(\boldsymbol{x}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\boldsymbol{x})]\right)^2\right]}_{\textbf{Variance}} + \underbrace{\sigma^2}_{\textbf{Irreducible Noise}}$$

- High Bias $\implies$ Underfitting (model too rigid).
- High Variance $\implies$ Overfitting (model captures sample fluctuations).

#### 5. Model Selection Criteria
- **Akaike Information Criterion (AIC):** $\text{AIC} = 2k - 2\log \hat{L}$ (asymptotically selects predictive model).
- **Bayesian Information Criterion (BIC):** $\text{BIC} = k\log N - 2\log \hat{L}$ (consistent; penalizes complexity $k$ more strongly as $N \to \infty$).

---

### Chapter 9: Linear Regression

> **Companion Notebook:** [`Ch09_Linear_Regression.ipynb`](Ch09_Linear_Regression.ipynb) &bull; 12 cells &bull; Exercises 9.1 to 9.4

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 9 examines linear regression as the quintessential supervised learning problem, synthesizing linear algebra, analytic geometry, matrix calculus, and Bayesian probability into a unified predictive framework. It begins with Maximum Likelihood estimation, showing that minimizing squared errors leads to the Normal equations, and illustrates through analytic geometry that predictions represent orthogonal projections of the target vector onto the subspace spanned by the feature columns ($\boldsymbol{\Phi}^\top \boldsymbol{e} = \boldsymbol{0}$). It expands into regularized MAP Ridge regression, analyzing parameter shrinkage along SVD singular directions and quantifying effective degrees of freedom. Finally, the chapter fully develops Bayesian Linear Regression, deriving closed-form Gaussian posterior parameter updates and posterior predictive distributions that cleanly partition total forecast variance into irreducible aleatoric data noise and epistemic parameter uncertainty.

#### 1. Maximum Likelihood Linear Regression
Model: $y = \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\theta} + \epsilon, \; \epsilon \sim \mathcal{N}(0, \sigma^2)$. Feature matrix $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$.
- **Normal Equations:**

$$\boldsymbol{\Phi}^\top \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^\top \boldsymbol{y} \implies \boldsymbol{\theta}_{\text{ML}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y}$$

- **Noise Variance Estimate:** $\sigma_{\text{ML}}^2 = \frac{1}{N}\|\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{\theta}_{\text{ML}}\|^2$.
- **Orthogonal Projector Perspective:** The prediction vector $\hat{\boldsymbol{y}} = \boldsymbol{\Phi}\boldsymbol{\theta}_{\text{ML}} = \boldsymbol{P}\boldsymbol{y}$ where $\boldsymbol{P} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top$. The residual $\boldsymbol{e} = \boldsymbol{y} - \hat{\boldsymbol{y}}$ satisfies $\boldsymbol{\Phi}^\top \boldsymbol{e} = \boldsymbol{0}$.

#### 2. MAP Ridge Regression
With prior $\boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{0}, b^2 \boldsymbol{I})$:

$$\boldsymbol{\theta}_{\text{MAP}} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \boldsymbol{I})^{-1}\boldsymbol{\Phi}^\top \boldsymbol{y}, \quad \lambda = \frac{\sigma^2}{b^2}$$

- **SVD Shrinkage:** Let $\boldsymbol{\Phi} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top$. The MAP prediction contracts along each principal mode by a shrinkage factor:

$$\hat{\boldsymbol{y}}_{\text{MAP}} = \sum_{i=1}^M \boldsymbol{u}_i \left(\frac{\sigma_i^2}{\sigma_i^2 + \lambda}\right) \boldsymbol{u}_i^\top \boldsymbol{y}$$

- **Effective Degrees of Freedom:** $\text{df}(\lambda) = \sum_{i=1}^M \frac{\sigma_i^2}{\sigma_i^2 + \lambda} = \mathrm{tr}\left(\boldsymbol{\Phi}(\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \boldsymbol{I})^{-1}\boldsymbol{\Phi}^\top\right)$.

#### 3. Bayesian Linear Regression
- **Prior:** $p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{m}_0, \boldsymbol{S}_0)$.
- **Likelihood:** $p(\boldsymbol{y} \mid \boldsymbol{\Phi}, \boldsymbol{\theta}, \sigma^2) = \mathcal{N}(\boldsymbol{\Phi}\boldsymbol{\theta}, \sigma^2 \boldsymbol{I})$.
- **Exact Posterior Distribution:** $p(\boldsymbol{\theta} \mid \mathcal{D}) = \mathcal{N}(\boldsymbol{m}_N, \boldsymbol{S}_N)$ where:

$$\boldsymbol{S}_N^{-1} = \boldsymbol{S}_0^{-1} + \sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$$

$$\boldsymbol{m}_N = \boldsymbol{S}_N\left(\boldsymbol{S}_0^{-1}\boldsymbol{m}_0 + \sigma^{-2}\boldsymbol{\Phi}^\top \boldsymbol{y}\right)$$

- **Posterior Predictive Distribution for a new query point $\boldsymbol{x}_{\ast}$:**

$$p(y_{\ast} \mid \boldsymbol{x}_{\ast}, \mathcal{D}) = \mathcal{N}\left(\boldsymbol{\phi}(\boldsymbol{x}_{\ast})^\top \boldsymbol{m}_N, \; \sigma_{\ast}^2(\boldsymbol{x}_{\ast})\right)$$

$$\sigma_{\ast}^2(\boldsymbol{x}_{\ast}) = \underbrace{\sigma^2}_{\textbf{Aleatoric Noise}} + \underbrace{\boldsymbol{\phi}(\boldsymbol{x}_{\ast})^\top \boldsymbol{S}_N \boldsymbol{\phi}(\boldsymbol{x}_{\ast})}_{\textbf{Epistemic Parameter Uncertainty}}$$

> As sample size $N \to \infty$, parameter covariance $\boldsymbol{S}_N \to \boldsymbol{0}$, so epistemic uncertainty vanishes and total predictive variance converges to irreducible noise $\sigma^2$.

---

### Chapter 10: Dimensionality Reduction with Principal Component Analysis (PCA)

> **Companion Notebook:** [`Ch10_PCA.ipynb`](Ch10_PCA.ipynb) &bull; 15 cells &bull; Exercises 10.1 to 10.5

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 10 investigates Principal Component Analysis (PCA) as the foundational unsupervised algorithm for dimensionality reduction, feature compression, and data visualization. The chapter demonstrates that two seemingly distinct mathematical objectives—maximizing the projected variance of data points along orthogonal directions, and minimizing the average squared reconstruction error between original vectors and low-rank reconstructions—yield the exact same linear algebraic solution: an eigendecomposition of the sample covariance matrix $\boldsymbol{S}$. It connects PCA to SVD and the Eckart–Young–Mirsky low-rank theorem, introduces the dual Gram-matrix trick $(\mathcal{O}(N^3) \text{ vs } \mathcal{O}(D^3))$ for high-dimensional regimes where $D \gg N$, and concludes with Probabilistic PCA (PPCA), framing dimensionality reduction as a generative latent variable model solved via maximum likelihood.

#### 1. Two Complementary Perspectives
Given centered dataset $\tilde{\boldsymbol{x}}_n = \boldsymbol{x}_n - \boldsymbol{\mu} \in \mathbb{R}^D$ with sample covariance matrix:

$$\boldsymbol{S} = \frac{1}{N}\sum_{n=1}^N \tilde{\boldsymbol{x}}_n \tilde{\boldsymbol{x}}_n^\top$$

1. **Maximum Projected Variance:** Find orthonormal axes $\boldsymbol{b}_1, \dots, \boldsymbol{b}_M$ maximizing variance:

$$\max_{\Vert\boldsymbol{b}_1\Vert=1} \boldsymbol{b}_1^\top \boldsymbol{S}\boldsymbol{b}_1 \implies \boldsymbol{S}\boldsymbol{b}_1 = \lambda_1 \boldsymbol{b}_1$$

2. **Minimum Reconstruction Error:** Find $M$-dimensional subspace minimizing average squared Euclidean distance between data points $\tilde{\boldsymbol{x}}_n$ and reconstructions $\hat{\boldsymbol{x}}_n = \boldsymbol{B}\boldsymbol{z}_n$:

$$\min_{\boldsymbol{B}} \frac{1}{N}\sum_{n=1}^N \Vert\tilde{\boldsymbol{x}}_n - \boldsymbol{B}\boldsymbol{B}^\top \tilde{\boldsymbol{x}}_n\Vert^2 \implies \text{Error} = \sum_{j=M+1}^D \lambda_j$$

Both formulations lead to the **exact same eigenvalue problem**: Choose the $M$ eigenvectors of $\boldsymbol{S}$ corresponding to the $M$ largest eigenvalues.

#### 2. High-Dimensional Dual PCA ($D \gg N$)
When $D \gg N$ (e.g. genomic sequences or high-resolution images, $D=12,000, N=100$), forming $\boldsymbol{S} \in \mathbb{R}^{D \times D}$ and computing its eigenvalues costs $\mathcal{O}(D^3)$ flops and gigabytes of RAM.
- **Dual Formulation:** Form the Gram matrix $\boldsymbol{K} = \frac{1}{N}\tilde{\boldsymbol{X}}\tilde{\boldsymbol{X}}^\top \in \mathbb{R}^{N \times N}$.
- Compute eigendecomposition of $\boldsymbol{K}$ in $\mathcal{O}(N^3)$: $\boldsymbol{K}\boldsymbol{a}_m = \lambda_m \boldsymbol{a}_m$.
- Recover true high-dimensional principal axes in $\mathbb{R}^D$:

$$\boldsymbol{b}_m = \frac{1}{\sqrt{N\lambda_m}}\tilde{\boldsymbol{X}}^\top \boldsymbol{a}_m$$

#### 3. Probabilistic PCA (PPCA)
Latent variable model: $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}_M)$, observation model $\boldsymbol{x} \mid \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{W}\boldsymbol{z} + \boldsymbol{\mu}, \sigma^2 \boldsymbol{I}_D)$.
- Exact ML parameter solutions:

$$\boldsymbol{W}_{\text{ML}} = \boldsymbol{U}_M (\boldsymbol{\Lambda}_M - \sigma^2 \boldsymbol{I})^{1/2}\boldsymbol{R}, \quad \sigma_{\text{ML}}^2 = \frac{1}{D-M}\sum_{j=M+1}^D \lambda_j$$

where $\boldsymbol{R} \in \mathrm{SO}(M)$ is an arbitrary rotation matrix.

---

### Chapter 11: Density Estimation with Gaussian Mixture Models (GMM)

> **Companion Notebook:** [`Ch11_GMM.ipynb`](Ch11_GMM.ipynb) &bull; 11 cells &bull; Exercises 11.1 to 11.4

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 11 explores Gaussian Mixture Models (GMMs) for multi-modal density estimation and soft clustering, addressing the challenge of modeling complex data distributions that cannot be captured by a single Gaussian. The chapter reveals why direct maximum likelihood estimation fails due to non-convex log-sum objectives and singular covariance pathologies, motivating the introduction of discrete latent indicator variables and the general Expectation-Maximization (EM) algorithm. Through Jensen's inequality and the Evidence Lower Bound (ELBO), it derives the iterative E-step (computing soft responsibilities) and closed-form M-step parameter updates. Furthermore, it mathematically proves that standard K-Means clustering is the hard-assignment zero-variance limit ($\sigma^2 \to 0$) of the EM algorithm, and examines covariance parameterization constraints and model selection using AIC and BIC.

#### 1. Mixture Formulation & Log-Likelihood Pathology
The GMM expresses arbitrary multi-modal densities as convex combinations of Gaussians:

$$p(\boldsymbol{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad \sum_{k=1}^K \pi_k = 1, \; \pi_k \ge 0$$

- **Log-Likelihood:** $\log p(\boldsymbol{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left(\sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right)$.
- **Pathology:** Sum inside the logarithm prevents closed-form stationary equations, and singularities occur when a component's covariance collapses onto a single point ($\det(\boldsymbol{\Sigma}_k) \to 0, \log L \to \infty$).

#### 2. Latent Variables & The EM Algorithm
Introduce binary latent indicator vector $\boldsymbol{z_n} \in \{0, 1\}^K$, where $z_{nk} = 1$ if sample $n$ belongs to component $k$.
- **Expectation Step (E-step):** Evaluate posterior responsibilities $\gamma_{nk} = p(z_{nk}=1 \mid \boldsymbol{x}_n, \boldsymbol{\theta})$:

$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

- **Maximization Step (M-step):** Update parameters using weighted averages:
  - Effective cluster size: $N_k = \sum_{n=1}^N \gamma_{nk}$
  - Mixture weights: $\pi_k^{\text{new}} = \frac{N_k}{N}$
  - Means:

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}\boldsymbol{x}_n$$

  - Covariances:

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{\text{new}})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{\text{new}})^\top$$

#### 3. Connection to K-Means
When covariances are constrained to spherical isotropic $\boldsymbol{\Sigma}_k = \sigma^2 \boldsymbol{I}$ and the variance limit $\sigma^2 \to 0$ is taken, responsibilities harden:

$$\gamma_{nk} \to \begin{cases} 1 & \text{if } k = \arg\min_j \|\boldsymbol{x}_n - \boldsymbol{\mu}_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$

Standard K-Means is precisely the hard-assignment zero-variance limit of EM!

---

### Chapter 12: Classification with Support Vector Machines (SVM)

> **Companion Notebook:** [`Ch12_SVM.ipynb`](Ch12_SVM.ipynb) &bull; 16 cells &bull; Exercises 12.1 to 12.4

> 📖 **Chapter Overview & Core Narrative:**  
> Chapter 12 develops Support Vector Machines (SVMs) as the canonical maximum-margin approach to binary classification, synthesizing hyperplane geometry, convex optimization, Lagrangian duality, and functional analysis. It formulates the hard-margin primal problem as finding the separating hyperplane that maximizes geometric margin $\frac{2}{\|\boldsymbol{w}\|}$, and derives the Lagrangian dual quadratic program, using KKT complementary slackness to prove that the optimal decision boundary depends exclusively on a sparse subset of data points sitting directly on the margin—the support vectors. The chapter extends this formulation to non-separable data via slack variables, establishing the equivalence between soft-margin SVMs and regularized Hinge loss minimization. Finally, it presents the Kernel Trick, leveraging Mercer's theorem and positive semi-definite Gram matrices to implicitly project inputs into infinite-dimensional reproducing kernel Hilbert spaces (e.g., via RBF kernels) while computing exclusively in the input space.

#### 1. Hard-Margin SVM (Linearly Separable Case)
Given binary classification dataset $\mathcal{D} = \{(\boldsymbol{x_n}, y_n)\}_{n=1}^N$ with labels $y_n \in \{-1, +1\}$:
- **Canonical Hyperplane:** $\min_n y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b) = 1$. The margin width between classes is $\frac{2}{\|\boldsymbol{w}\|}$.
- **Primal Quadratic Program:**

$$\min_{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^2 \quad \text{subject to} \quad y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b) \ge 1 \quad (\forall n=1,\dots,N)$$

- **Lagrangian Dual QP:**

$$\max_{\boldsymbol{\alpha}} \sum_{n=1}^N \alpha_n - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m (\boldsymbol{x}_n^\top \boldsymbol{x}_m) \quad \text{s.t.} \quad \alpha_n \ge 0, \; \sum_{n=1}^N \alpha_n y_n = 0$$

- **KKT Complementary Slackness:** $\alpha_n [y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b) - 1] = 0$.
  - Points with $\alpha_n = 0$ lie strictly outside the margin.
  - Points with $\alpha_n > 0$ lie **exactly on the margin** ($y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b) = 1$). These are the **Support Vectors**!
- Optimal weights: $\boldsymbol{w}^\star = \sum_{n \in \text{SV}} \alpha_n y_n \boldsymbol{x}_n$.

#### 2. Soft-Margin SVM (Non-Separable Case)
Introduce slack variables $\xi_n \ge 0$ penalizing margin violations:
- **Primal QP:** $\min_{\boldsymbol{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{n=1}^N \xi_n$ s.t. $y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b) \ge 1 - \xi_n, \; \xi_n \ge 0$.
- **Equivalent Hinge Loss Form:** $\min_{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_{n=1}^N \max(0, 1 - y_n(\boldsymbol{w}^\top \boldsymbol{x}_n + b))$.
- **Dual QP:** Exactly identical to hard-margin dual, with box constraints:

$$0 \le \alpha_n \le C \quad (\forall n=1,\dots,N), \quad \sum_{n=1}^N \alpha_n y_n = 0$$

#### 3. The Kernel Trick & Mercer's Theorem
Replace inner products $\boldsymbol{x}_n^\top \boldsymbol{x}_m$ with kernel functions $k(\boldsymbol{x}_n, \boldsymbol{x}_m) = \langle \boldsymbol{\phi}(\boldsymbol{x}_n), \boldsymbol{\phi}(\boldsymbol{x}_m) \rangle$.
- **Mercer's Condition:** A symmetric kernel $k(\boldsymbol{x}, \boldsymbol{z})$ is valid iff its Gram matrix $\boldsymbol{K} \in \mathbb{R}^{N \times N}$ ($K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$) is positive semi-definite ($\boldsymbol{K} \succeq 0$) for any dataset.
- **Common Kernels:**
  - Polynomial: $k(\boldsymbol{x}, \boldsymbol{z}) = (\boldsymbol{x}^\top \boldsymbol{z} + c)^d$.
  - Gaussian / RBF: $k(\boldsymbol{x}, \boldsymbol{z}) = \exp(-\gamma \|\boldsymbol{x} - \boldsymbol{z}\|^2)$ (corresponds to an infinite-dimensional feature map).
- **Non-Linear Decision Function:**

$$f(\boldsymbol{x}_{\ast}) = \text{sign}\left(\sum_{n \in \text{SV}} \alpha_n y_n k(\boldsymbol{x}_n, \boldsymbol{x}_{\ast}) + b\right)$$

---

## ⚡ Master Reference Cards & Rapid-Fire Tables

---

### A. Matrix Calculus Identities

| Expression $f(\boldsymbol{x})$ or $f(\boldsymbol{A})$ | Derivative with respect to vector $\boldsymbol{x}$ or matrix $\boldsymbol{A}$ | Notes |
|:---|:---|:---|
| $\boldsymbol{a}^\top \boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} f = \boldsymbol{a}$ | Linear vector form |
| $\boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}$ | $\nabla_{\boldsymbol{x}} f = (\boldsymbol{A} + \boldsymbol{A}^\top)\boldsymbol{x}$ | Becomes $2\boldsymbol{A}\boldsymbol{x}$ if $\boldsymbol{A} = \boldsymbol{A}^\top$ |
| $\|\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}\|^2$ | $\nabla_{\boldsymbol{x}} f = 2\boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})$ | Least-squares objective |
| $\mathrm{tr}(\boldsymbol{A}\boldsymbol{B})$ | $\nabla_{\boldsymbol{A}} f = \boldsymbol{B}^\top$ | Trace derivative |
| $\mathrm{tr}(\boldsymbol{A}^\top \boldsymbol{B} \boldsymbol{A} \boldsymbol{C})$ | $\nabla_{\boldsymbol{A}} f = \boldsymbol{B}\boldsymbol{A}\boldsymbol{C} + \boldsymbol{B}^\top \boldsymbol{A}\boldsymbol{C}^\top$ | Quadratic trace form |
| $\log \det \boldsymbol{A}$ | $\nabla_{\boldsymbol{A}} f = \boldsymbol{A}^{-\top} = (\boldsymbol{A}^{-1})^\top$ | For positive-definite $\boldsymbol{A} \succ 0$ |
| $\det(\boldsymbol{A})$ | $\nabla_{\boldsymbol{A}} f = \det(\boldsymbol{A})\boldsymbol{A}^{-\top}$ | Jacobi's formula |

---

### B. Matrix Decompositions Comparison Matrix

| Decomposition | Form | Matrix Requirements | Geometric / Computational Meaning | Primary ML Applications |
|:---|:---:|:---|:---|:---|
| **Cholesky** | $\boldsymbol{A} = \boldsymbol{L}\boldsymbol{L}^\top$ | Symmetric Positive-Definite ($\boldsymbol{A} \succ 0$) | "Square root" of covariance matrix | Sampling multivariate Gaussians, solving Normal equations |
| **QR** | $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{R}$ | Any $M \times N$ matrix | Gram-Schmidt orthonormalization | Numerically stable linear least squares |
| **Eigendecomp** | $\boldsymbol{A} = \boldsymbol{P}\boldsymbol{D}\boldsymbol{P}^{-1}$ | Square diagonalizable ($N$ linearly independent eigenvectors) | Decoupling into invariant eigen-directions | Markov chains, graph Laplacians, stability analysis |
| **Spectral** | $\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^\top$ | Real Symmetric ($\boldsymbol{A} = \boldsymbol{A}^\top$) | Orthogonal axes scaling | Covariance decomposition, Kernel PCA, spectral clustering |
| **SVD** | $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top$ | **Any** real matrix $M \times N$ | Rotation $\to$ Scaling $\to$ Rotation | PCA, low-rank compression, pseudoinverse, latent semantic analysis |

---

### C. Probability Distributions Quick Sheet

| Distribution | Support / Domain | Parameters | Mean $\mathbb{E}[X]$ | Variance $\mathbb{V}[X]$ | Conjugate Prior |
|:---|:---|:---|:---|:---|:---|
| **Bernoulli** | $x \in \{0, 1\}$ | $\mu \in [0, 1]$ | $\mu$ | $\mu(1 - \mu)$ | Beta |
| **Binomial** | $k \in \{0, \dots, N\}$ | $N \in \mathbb{N}, \mu \in [0, 1]$ | $N\mu$ | $N\mu(1 - \mu)$ | Beta |
| **Multinomial** | $\boldsymbol{x} \in \mathbb{N}^K, \sum x_k = N$ | $\boldsymbol{\pi}, \sum \pi_k = 1$ | $N\boldsymbol{\pi}$ | $\mathrm{Cov}(x_i, x_j) = N\pi_i(\delta_{ij} - \pi_j)$ | Dirichlet |
| **Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$** | $\boldsymbol{x} \in \mathbb{R}^D$ | $\boldsymbol{\mu} \in \mathbb{R}^D, \boldsymbol{\Sigma} \succ 0$ | $\boldsymbol{\mu}$ | $\boldsymbol{\Sigma}$ | Gaussian (mean), Wishart (precision) |
| **Beta** | $x \in [0, 1]$ | $\alpha > 0, \beta > 0$ | $\frac{\alpha}{\alpha + \beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | Conjugate to Binomial/Bernoulli |
| **Dirichlet** | $\boldsymbol{x} \in \Delta^K$ (simplex) | $\boldsymbol{\alpha} \in \mathbb{R}_{>0}^K$ | $\frac{\alpha_k}{\sum \alpha_j}$ | $\frac{\tilde{\alpha}_k(1-\tilde{\alpha}_k)}{\alpha_0 + 1}$ | Conjugate to Multinomial/Categorical |

---

### D. Central ML Algorithms Comparison Matrix

| Property | Linear Regression (Ch 9) | PCA (Ch 10) | GMM (Ch 11) | SVM (Ch 12) |
|:---|:---|:---|:---|:---|
| **Problem Type** | Supervised continuous prediction | Unsupervised dimensionality reduction | Unsupervised density estimation / clustering | Supervised binary classification |
| **Loss / Objective** | $\frac{1}{2}\|\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{\theta}\|^2 + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ | Maximize projected variance $\boldsymbol{b}^\top \boldsymbol{S}\boldsymbol{b}$ | Maximize log-likelihood $\sum \log p(\boldsymbol{x}_n \mid \boldsymbol{\theta})$ | Maximize margin $\frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum \xi_n$ |
| **Exact Solution?** | Closed-form Normal equations | Closed-form Eigendecomposition / SVD | Iterative EM algorithm (local optima) | Convex Quadratic Program (global optimum) |
| **Primal / Dual** | Primal ($\boldsymbol{\Phi}^\top \boldsymbol{\Phi} \in \mathbb{R}^{M \times M}$) or Dual GP | Primal ($\boldsymbol{S} \in \mathbb{R}^{D \times D}$) vs Dual Gram ($\boldsymbol{K} \in \mathbb{R}^{N \times N}$) | ELBO bound ascent | Primal ($D$ weights) vs Dual ($N$ Lagrange multipliers $\alpha_n$) |
| **Key Assumptions** | Gaussian additive noise, linearity | Linear subspace, orthogonal axes | Mixture of Gaussian components | Linear separability in feature space $\mathcal{H}$ |
| **Notebook Link** | [`Ch09_Linear_Regression.ipynb`](Ch09_Linear_Regression.ipynb) | [`Ch10_PCA.ipynb`](Ch10_PCA.ipynb) | [`Ch11_GMM.ipynb`](Ch11_GMM.ipynb) | [`Ch12_SVM.ipynb`](Ch12_SVM.ipynb) |

---

<p align="center">
  <em>Mathematics for Machine Learning Comprehensive Cheatsheet &bull; Authored for MSc Artificial Intelligence</em>
</p>
