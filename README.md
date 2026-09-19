<p align="center">
  <h1 align="center">🎓 MSc Artificial Intelligence — Coursework</h1>
  <p align="center">
    <strong>Jomo Kenyatta University of Agriculture and Technology (JKUAT)</strong><br>
    <em>A curated collection of assignments, projects, and research across the MSc AI programme</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Degree-MSc_Artificial_Intelligence-blue?style=for-the-badge" alt="Degree"/>
    <img src="https://img.shields.io/badge/University-JKUAT-green?style=for-the-badge" alt="University"/>
    <img src="https://img.shields.io/badge/Status-In_Progress:_Thesis_Ongoing-orange?style=for-the-badge" alt="Status"/>
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules & Research](#modules--research)
  - [1. Semester 1 — Foundations](#1-semester-1--foundations)
    - [AI Algos](#ai-algos) — Core AI search strategies, fuzzy inference systems & agent pathfinding in unknown environments
    - [AI Applications](#ai-applications) — Econometric causal inference study (*The Marginal Utility of Police Force*) based on the Marginal Utility repository
    - [Foundations of AI](#foundations-of-ai) — Foundational optimisation paradigms & Mixed-Integer Linear Programming (MILP)
    - [Mathematics for AI (Maths4AI)](#mathematics-for-ai-maths4ai) — Mathematical rigor in linear algebra, multivariable calculus & probability theory
    - [ML Foundations](#ml-foundations) — Supervised & unsupervised learning, regularisation, Perceptrons, CATs & examination papers (Sec A & B)
    - [Python 4 AI](#python-4-ai) — Python programming fundamentals, data manipulation (Pandas/NumPy) & assignment solutions
  - [2. Semester 2 — Advanced Modules](#2-semester-2--advanced-modules)
    - [Computer Vision](#computer-vision) — Fourier spectral filtering, Canny/Hough/Harris feature detection & Seq2Seq image captioning
    - [Deep Learning](#deep-learning) — MLPs, CNNs, RNN vs. GRU comparison, Hamming networks, Kohonen SOM & WTA competitive learning
    - [Natural Language Processing](#natural-language-processing) — Marian NMT full-parameter translation pipeline for Kamba (*Kikamba*) to English
    - [Robotics](#robotics) — SLAM simulation with LiDAR ray-casting, odometry & log-odds occupancy mapping
    - [Ethics & Governance in AI](#ethics--governance-in-ai) — AI sovereignty, global governance architectures (EU AI Act, US EOs, BRICS) & ICRIER IPCIDE synthesis
    - [Research Methodology](#research-methodology) — Culturally-adapted conversational AI for youth mental health (CompanionAI Clean Architecture prototype)
  - [3. Thesis — MSc Research Proposal](#3-thesis--msc-research-proposal) — Stress testing explainable AI (XAI) in non-linear credit scoring models (SHAP, LIME, EAS, and counterfactual analysis)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Author](#author)
- [License](#license)

---

## 🔭 Overview

This repository houses the complete academic coursework, laboratory implementations, and research artefacts produced during my **MSc in Artificial Intelligence** at **Jomo Kenyatta University of Agriculture and Technology (JKUAT)**.

The academic trajectory is organised into three distinct phases:
1. **Semester 1 (Foundations)**: Core mathematical and computational pillars — from search heuristics, fuzzy logic, and Mixed-Integer Linear Programming to foundational machine learning, perceptrons, and scientific Python.
2. **Semester 2 (Advanced AI Disciplines)**: Specialised technical domains spanning deep learning architectures, computer vision, sequence-to-sequence modelling, neural machine translation for low-resource languages, autonomous SLAM robotics, AI ethics & governance policy, and research methodology.
3. **Thesis (Ongoing Research)**: Postgraduate research proposal on stress-testing post-hoc Explainable AI (XAI) methods under input perturbations in credit scoring models, adhering to EU AI Act high-risk compliance frameworks.

---

## 📁 Repository Structure

```
Msc-AI-Coursework/
├── .venv/                              # Python virtual environment (local / gitignored)
├── Sem 1/                              # Semester 1 — Foundational AI modules
│   ├── AI Algos/                       #   AI search algorithms, fuzzy logic & agent pathfinding
│   ├── AI Applications/                #   Causal inference paper (Marginal Utility repo)
│   ├── Foundations of AI/              #   Optimisation theory & Mixed-Integer Linear Programming
│   ├── Maths4AI/                       #   Mathematics for AI continuous assessment
│   ├── ML Foundations/                 #   Machine learning exams (Sec A & B), CATs & perceptron
│   └── Python 4 AI/                   #   Python programming, data analysis & assignment solutions
├── Sem 2/                              # Semester 2 — Advanced AI modules
│   ├── Computer Vision/                #   Spectral filtering, feature detection & Seq2Seq captioning
│   ├── Deep Learning/                  #   MLP, CNN, RNN/GRU, Hamming networks, Kohonen SOM & WTA
│   ├── Natural Lang Processing/        #   Kamba–English Neural Machine Translation pipeline
│   ├── Robotics/                       #   SLAM-based mapping robot simulation & LiDAR ray-casting
│   ├── Ethics & Governance in AI/      #   AI policy, EU AI Act, sovereignty & ICRIER synthesis
│   └── Research Methodology/           #   CompanionAI mental health research proposal & prototype
├── Thesis/                             # MSc Thesis — Stress testing XAI in credit scoring
└── .gitignore
```

---

## 📚 Modules & Research

### 1. Semester 1 — Foundations

> *Core AI algorithms, mathematical optimisation, foundational machine learning, and scientific Python*

#### AI Algos

Exploration of state-space search, heuristic pathfinding, partially observable environments, and fuzzy inference systems.

| File | Topic / Description |
|:--|:--|
| `AI_Algos_Assigment.ipynb` | AI search algorithms — Assignments 1–4 (state-space search, heuristics) |
| `AI ALgos Exam.pdf` | AI Algorithms comprehensive examination paper |
| `Agent Search in Unknown Environments.pdf` | Formal paper on agent-based search strategies in unknown, partially observable environments |
| `fuzzy_logic.py` | Standalone fuzzy inference engine with custom triangular and trapezoidal membership functions |
| `Tipping_Soln.ipynb` | Fuzzy logic tipping control system with multiple rule bases and membership tuning |
| `Tipping_Soln.pdf` | Compiled analysis and visualisations of the fuzzy tipping solution |
| `Random Unknown.py` | A* pathfinding and stochastic search with dynamic obstacle avoidance on a 2D grid |
| `Search Unknown.py` | Exploration algorithms for agents operating under partial observability |

#### AI Applications

Econometric and applied AI research on causal inference and public policy analysis.

| File | Topic / Description |
|:--|:--|
| `AI Paper Revised.pdf` | **Research Paper:** *The Marginal Utility of Police Force: A Causal Analysis Paper* |

> 📌 **Project Background & Repository:**
>
> This research paper is based on the **[Marginal Utility](https://github.com/KeSeaman/Marginal-Utility)** repository. It investigates whether increasing investment in police funding causally reduces violent crime rates across **2,624 US municipal cities** using observational and budgetary data.
>
> **Methodology & Findings:**
> - **Econometric Design:** Implements a quasi-experimental **Difference-in-Differences (DiD)** design combined with **Propensity Score Matching (PSM)** (1:1 Nearest Neighbour within a 0.25 SD caliper) and **Doubly Robust Estimation** combining PSM weighting with outcome regression adjustment.
> - **Empirical Results:** Contrary to the standard deterrence hypothesis, cities with large police budget increases exhibited a greater increase in violent crime (+344 incidents per 100,000 residents from 2015 to 2019) compared to matched control cities.
> - **Placebo & Sensitivity Checks:** A placebo test on property crime (+1181.84 incidents) and **Rosenbaum bounds sensitivity analysis** ($\Gamma = 1.5$) demonstrate that these findings are heavily confounded by broader macroeconomic booms and gentrification dynamics, which simultaneously swell municipal tax revenues for policing and increase the asset base vulnerable to property crimes.

#### Foundations of AI

Analytical and computational optimisation for combinatorial decision-making.

| File | Topic / Description |
|:--|:--|
| `Foundations of AI Optimization Report.pdf` | Comprehensive technical report on classical and heuristic optimisation paradigms in AI |
| `MILP.ipynb` | **Mixed-Integer Linear Programming (MILP)** — Warehouse supply chain routing and resource allocation using PuLP |

#### Mathematics for AI (Maths4AI)

Mathematical foundations underpinning machine learning and probabilistic reasoning.

| File | Topic / Description |
|:--|:--|
| `Kinyua_Seaman_ CAT_Maths4AI.pdf` | Continuous Assessment Test (CAT) covering vector spaces, matrix factorisations, multivariate gradient calculus, and probability distributions |

#### ML Foundations

Core machine learning theory, supervised learning benchmarks, margin classifiers, and empirical evaluation (Course code: **ICS 3305**).

| File | Topic / Description |
|:--|:--|
| `ICS3305 - ML Foundations.pdf` | **Machine Learning Foundations Examination — Section A:** Regularisation (L1/L2), overfitting mitigation, and statistical learning |
| `ICS3305 - ML Foundations Sec B.pdf` | **Machine Learning Foundations Examination — Section B:** Support Vector Machines (SVM), kernel trick, and sensor-based activity recognition |
| `Kinyua_Seaman_ML.pdf` | **Machine Learning CAT 2 Submission:** Supervised vs. unsupervised learning theory, customer segmentation & loss functions |
| `Kinyua Seaman ML_CAT.pdf` | Machine Learning Continuous Assessment Test (CAT 1) |
| `Perceptron Assignment.pdf` | Theoretical derivation and geometric proofs of the Rosenblatt Perceptron convergence theorem |
| `perceptron.ipynb` | Single-layer Perceptron implementation from scratch with decision boundary visualisation |

#### Python 4 AI

High-performance scientific Python programming, exploratory data analysis, and numerical processing.

| File | Topic / Description |
|:--|:--|
| `Python4AI.ipynb` | Advanced Python programming: NumPy array operations, Pandas dataframe transformations, and Matplotlib plotting |
| `K_Seaman_Q2.ipynb` | Practical data analysis assignments and predictive modelling exercises |
| `KENTemp.csv` | Historical Kenya temperature time-series dataset utilised for trend analysis |
| `sales.csv` | Multi-category commercial sales transactions dataset for exploratory analysis |

---

### 2. Semester 2 — Advanced Modules

> *Specialised engineering disciplines: deep architectures, visual computing, translation, robotics, governance, and research formulation*

#### Computer Vision

Image processing, spectral analysis, feature extraction, and sequence-to-sequence vision-language architectures.

| Notebook / File | Topic / Description |
|:--|:--|
| `Solution_CV.ipynb` | Fourier transform (FFT) noise removal, Canny edge detection, Hough transform, and Harris corner detection |
| `Assignment_2.ipynb` | Advanced computer vision techniques and image transformation pipelines |
| `Assignment 1.pdf` | Computer Vision Assignment 1 solutions |
| `Term_Paper.ipynb` / `Term Paper.pdf` | **Term Paper:** *Comparative Analysis of Seq2Seq Architectures* — Case study of Google's Show-and-Tell image captioning framework |
| `Kinyua_Seaman_Term_Paper.pdf` | Compiled research term paper on encoder-decoder image captioning architectures |
| `report.pdf` / `report.tex` | Full laboratory report and experimental analysis |

**Key Techniques:** 2D Fast Fourier Transform (FFT) · Canny edge detection · Hough line transform · Harris corner detection · Encoder-Decoder architectures (CNN encoder + LSTM decoder)

---

#### Deep Learning

Comprehensive study of neural architectures (Course code: **ICS 3308**) — from multi-layer perceptrons to recurrent sequence models and unsupervised competitive learning.

| Notebook / File | Topic / Description |
|:--|:--|
| `Assignment_1_MLP_House_Prices.ipynb` | **Multi-Layer Perceptron (MLP)** for non-linear regression on the California Housing dataset |
| `Assignment_2_CNN_Image_Classification.ipynb` | **Convolutional Neural Network (CNN)** for image classification on CIFAR-10 |
| `Assignment_3_RNN_vs_GRU.ipynb` | **RNN vs. GRU** comparative analysis on sequential time-series forecasting |
| `hamming_network.py` / `.ipynb` | Hamming neural network implementation for binary pattern recognition |
| `self_organizing_map.py` / `.ipynb` | Kohonen Self-Organising Map (SOM) for unsupervised colour clustering and topological mapping |
| `winner_takes_all.py` / `.ipynb` | Winner-Takes-All (WTA) competitive learning network implementation |
| `Deep Learning Assignments.pdf` | Official deep learning coursework assignments specification (Assignments 1–4) |
| `ICS_3308_Deep_Learning.pdf` | Deep Learning course syllabus, lecture materials, and theoretical foundations |
| `ICS 3308 Deep Learning QP.pdf` | Examination question paper for ICS 3308 Deep Learning |
| `ICS_3308_Deep_Learning_QP_Solutions.ipynb` | Complete programming solutions for past examination questions |

<p align="center">
  <img src="Sem 2/Deep Learning/hamming_network_results.png" width="30%" alt="Hamming Network"/>
  <img src="Sem 2/Deep Learning/som_colour_evolution.png" width="30%" alt="SOM Colour Evolution"/>
  <img src="Sem 2/Deep Learning/wta_results.png" width="30%" alt="WTA Results"/>
</p>

---

#### Natural Language Processing

Development of an end-to-end Neural Machine Translation (NMT) pipeline for **Kamba (*Kikamba*)**, an under-resourced Bantu language spoken by ~4.7 million people in south-eastern Kenya.

| Notebook / File | Role / Description |
|:--|:--|
| `Machine_Translation.ipynb` | Local development notebook with data cleaning, tokenisation, and inference |
| `Machine_Translation_Colab_V1–V4.ipynb` | Iterative GPU training runs in Google Colab |
| `report.pdf` / `report.tex` | Research report: *Kamba–English Machine Translation: Neural Machine Translation for an Under-Resourced Bantu Language* |
| `pyproject.toml` | Environment dependencies and package configuration |

**Architecture & Setup:** Marian NMT (`Helsinki-NLP/opus-mt-en-bnt`) · Full-parameter fine-tuning · bf16 mixed-precision training · 51K parallel sentence pairs

<p align="center">
  <img src="Sem 2/Natural Lang Processing/metrics_plot.png" width="60%" alt="NLP Training Metrics"/>
</p>

---

#### Robotics

Perception and kinematics simulation implementing **Simultaneous Localisation and Mapping (SLAM)** for autonomous ground navigation:

- **Differential drive kinematics** with noisy odometry modelling
- **2D LiDAR ray-casting** rangefinder simulation with obstacle intersections
- **Log-odds occupancy grid mapping** driven by Bresenham's line algorithm

| File | Role / Description |
|:--|:--|
| `environment.py` | 2D simulated environment with polygon obstacles and ray-cast LiDAR sensors |
| `robot.py` | Differential drive kinematic state updates, control inputs, and odometry noise |
| `slam.py` | 2D occupancy grid with recursive log-odds Bayesian updates |
| `main.py` | Main execution loop, waypoint patrol sequence, and live map rendering |
| `Kinyua Seaman Report.pdf` | SLAM robotics engineering and kinematics report |
| `SLAM_Robotics_Presentation.pptx` | Project presentation slide deck |
| `visualization.html` | Interactive browser-based map rendering and playback |

```bash
# Run the robotics simulation
cd "Sem 2/Robotics"
uv run main.py
```

<p align="center">
  <img src="Sem 2/Robotics/plots/step_0000.png" width="23%" alt="Step 0"/>
  <img src="Sem 2/Robotics/plots/step_0350.png" width="23%" alt="Step 350"/>
  <img src="Sem 2/Robotics/plots/step_0700.png" width="23%" alt="Step 700"/>
  <img src="Sem 2/Robotics/plots/final_map.png" width="23%" alt="Final Map"/>
</p>

---

#### Ethics & Governance in AI

Critical analysis of global AI governance architectures, institutional mechanisms, ethical risk taxonomies, and the Global South perspective.

| File | Description | Format |
|:--|:--|:--|
| [`AI_G&E_Term_Paper.pdf`](Sem%202/Ethics%20%26%20Governance%20in%20AI/AI_G%26E_Term_Paper.pdf) | **Term Paper:** *AI Governance and Ethics: Frameworks and Challenges* | PDF (Compiled) |
| [`Ethics-and-Governance-of-AI-Synthesis.pdf`](Sem%202/Ethics%20%26%20Governance%20in%20AI/Ethics-and-Governance-of-AI-Synthesis.pdf) | IPCIDE Webinar Series Synthesis Report (ICRIER Prosus Centre, Oct 2025 – Jan 2026) | PDF (Source) |

##### 📑 Term Paper Core Structure

- **1. Introduction** — Geopolitical tensions, compute monopolies, and institutional fragmentation in generative AI.
- **2. Problem Statement** — Asymmetries in the global AI value chain, material resource exploitation, and Global South digital sovereignty deficits.
- **3. Overview of AI Governance** — Institutional mandates (UNESCO, OECD, G20, African Union) and international coordination.
- **4. Current Frameworks & Private Sector Governance** — Comparative analysis of the EU AI Act (risk-tiered model), US Executive Orders (market-driven benchmarks), and BRICS multilateral frameworks; corporate self-regulation limits and antitrust policies.
- **5. Ethical Considerations in AI** — Algorithmic bias, allocative harms, surveillance capitalism, cross-border data protection, model auditability, and strict supply chain accountability.
- **6. Future Considerations** — Operationalising the **SCAIS framework**, securing the sovereign AI stack (**KASE: Knowledge, Compute, Sovereign Cloud, Data**), and conceptualising AI as **Digital Public Infrastructure (DPI)**.
- **7. Conclusions and Recommendations** — Actionable policy roadmaps for developing economies balancing technological innovation with human rights.
- **8. References** — Comprehensive academic and multilateral policy bibliography.

---

#### Research Methodology

Formulation of an MSc research proposal investigating culturally-adapted conversational AI for Kenyan youth mental health (**CompanionAI**).

| File | Description |
|:--|:--|
| `Proposal_Kinyua_Seaman.pdf` / `.tex` | Full research proposal documentation and LaTeX source |
| `CompanionAI+Research+Programme.pdf` | Comprehensive research programme overview |
| `Presentation Skills Enhancement (1).odp` | Academic presentation skills and defence preparation deck |
| `blog.html` | Public science communication blog post |
| `ca_ai/` | **CompanionAI prototype** — Clean Architecture Python backend |

**Prototype Architecture (`ca_ai/`):**
```
ca_ai/
├── core/           # Domain logic: clinical safety rules, prompt builders, protocols
├── adapters/       # External service interfaces (mock adapters)
├── use_cases/      # Application business logic (chat interaction workflows)
├── api/            # FastAPI REST endpoints
├── infrastructure/ # System infrastructure & persistence
└── tests/          # Unit and integration test suites
```

---

### 3. Thesis — MSc Research Proposal

> *Postgraduate MSc Thesis Proposal — Stress testing explainability and decision alignment in regulated credit scoring models*

| File | Description | Format |
|:--|:--|:--|
| [`MSc_Thesis_Proposal.pdf`](Thesis/MSc_Thesis_Proposal.pdf) | **MSc Thesis Proposal:** *Stress Testing Explainable AI in Non-Linear Credit Scoring Models: A Framework for Evaluating Stability and Decision-Alignment of Feature Attribution Methods* | PDF (Compiled) |
| [`MSc Thesis Proposal Stress.md`](Thesis/MSc%20Thesis%20Proposal%20Stress.md) | Complete proposal manuscript (Abstract, Problem Statement, Research Questions, Literature Review, Methodology, Statistical Framework) | Markdown |
| `MSc_Thesis_Proposal.tex` | Full LaTeX source document for the formal thesis proposal | LaTeX |
| `deepseek_text_20260616_435c2e.txt` | Literature synthesis, reference notes, and technical citations | Text |

#### 🎯 Research Overview & Core Questions

Credit scoring systems increasingly deploy non-linear models (e.g., **XGBoost**, gradient boosting, TabNet) for superior predictive discrimination over linear models. However, regulatory frameworks — most notably the **European Union Artificial Intelligence Act**, which designates credit scoring as a **high-risk AI application** with 2026–2027 compliance mandates — require automated decisions to be stable, auditable, and actionable.

While post-hoc Explainable AI (XAI) methods like **SHAP** (Shapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) are widely adopted, current validation techniques fail to stress-test whether explanations remain consistent under realistic data perturbations.

```
                  ┌─────────────────────────────────┐
                  │   Credit Risk Benchmark Data    │
                  │ (Home Credit / Taiwan Default)  │
                  └────────────────┬────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
      ┌─────────────────────┐             ┌─────────────────────┐
      │ Logistic Regression │             │  XGBoost Classifier │
      │  (Linear Baseline)  │             │ (Non-Linear Model)  │
      └──────────┬──────────┘             └──────────┬──────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │    Feature Attribution Suite    │
                  │   SHAP  ·  LIME  ·  Hybrid      │
                  └────────────────┬────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
      ┌─────────────────────┐             ┌─────────────────────┐
      │ Perturbation Stress │             │   Counterfactual    │
      │   (Gaussian/Noise)  │             │  Generation (CCS)   │
      └──────────┬──────────┘             └──────────┬──────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │   Explanation Alignment Score   │
                  │           (EAS Metric)          │
                  └─────────────────────────────────┘
```

#### 🔬 Key Technical Contributions

1. **Explanation Alignment Score (EAS):** A novel metric quantifying explanation reliability by jointly measuring attribution stability under input perturbations and attribution sensitivity across decision boundary shifts.
2. **Constrained Counterfactual Generation (CCS):** An algorithmic framework assessing whether post-hoc attributions correctly highlight the minimal, actionable feature adjustments that change model classification.
3. **Empirical Benchmarking:** Extensive comparative evaluation across the **Home Credit Default Risk** and **Taiwan Credit Card Default** datasets, evaluating trade-offs between model non-linearity and explanatory fidelity.

---

## 🛠 Tech Stack

| Category | Technologies |
|:--|:--|
| **Languages** | Python 3.10+ |
| **Machine Learning & Deep Learning** | PyTorch, TensorFlow/Keras, scikit-learn, HuggingFace Transformers, XGBoost, Optuna |
| **Explainable AI (XAI)** | SHAP (Shapley Additive exPlanations), LIME, Custom Hybrid SHAP-LIME |
| **Computer Vision** | OpenCV, NumPy, Matplotlib, SciPy |
| **Natural Language Processing** | Marian NMT, SentencePiece, NLTK, HuggingFace Datasets |
| **Data Science & Econometrics** | Pandas, NumPy, SciPy, Statsmodels, CausalInference (DiD, PSM) |
| **Optimisation** | PuLP (MILP), scikit-fuzzy |
| **Robotics Simulation** | Custom differential kinematics & 2D LiDAR SLAM (NumPy, Matplotlib) |
| **Backend & Architecture** | FastAPI, Azure AI Search, Clean Architecture (CompanionAI) |
| **Package Management** | uv, pip |
| **Interactive Environments** | Jupyter Notebook, Google Colab |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/KeSeaman/Msc-AI-Coursework.git
cd Msc-AI-Coursework
```

### 2. Environment Setup

The repository utilizes a standard Python virtual environment. You can set it up using `uv` (recommended) or standard `venv`:

```bash
# Using uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # or install module dependencies

# Or using standard python venv
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Running Coursework & Simulations

```bash
# Launch Jupyter to explore notebooks across Sem 1 and Sem 2
jupyter notebook

# Run the SLAM Robotics Simulation (Semester 2)
cd "Sem 2/Robotics"
uv run main.py

# Explore the CompanionAI Prototype (Semester 2 Research Methodology)
cd "Sem 2/Research Methodology/ca_ai"
uv run pytest
```

> **Note on Large Datasets:** Extremely large datasets (such as raw video keypoints for Computer Vision) and local virtual environments (`.venv/`) are excluded from version control via `.gitignore`.

---

## 👤 Author

**Kinyua Seaman**
- GitHub: [@KeSeaman](https://github.com/KeSeaman)
- Institution: Jomo Kenyatta University of Agriculture and Technology (JKUAT)
- Programme: MSc Artificial Intelligence

---

## 📄 License

This repository is maintained for educational, academic, and portfolio demonstration purposes. Individual assignments and research materials remain subject to JKUAT academic integrity guidelines and relevant copyright policies.
