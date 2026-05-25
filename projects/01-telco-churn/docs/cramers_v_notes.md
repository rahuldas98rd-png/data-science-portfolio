# Cramér's V and the Measurement of Association — Notes & Reference

> **Personal study notes built during Project 01 (Telco Customer Churn).**
> Reusable for every classification project where feature-target association ranking matters.

---

## Table of Contents

1. [The Problem — Why Pearson Correlation Fails for Categoricals](#1-the-problem--why-pearson-correlation-fails-for-categoricals)
2. [The Chi-Square Test of Independence](#2-the-chi-square-test-of-independence)
3. [From Chi-Square to Cramér's V](#3-from-chi-square-to-cramérs-v)
4. [Interpretation Scale](#4-interpretation-scale)
5. [Bergsma's Bias Correction](#5-bergsmas-bias-correction)
6. [Limitations and Caveats](#6-limitations-and-caveats)
7. [Python Implementation](#7-python-implementation)
8. [The Broader Landscape — Association Measures Across Variable Types](#8-the-broader-landscape--association-measures-across-variable-types)
9. [Choosing the Right Tool — Decision Guide](#9-choosing-the-right-tool--decision-guide)
10. [Practical Workflow for Feature Ranking](#10-practical-workflow-for-feature-ranking)

---

## 1. The Problem — Why Pearson Correlation Fails for Categoricals

For two **numerical** variables, Pearson correlation gives a clean answer:

$$r = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i - \bar{x})^2 \cdot \sum_i(y_i - \bar{y})^2}}$$

It produces a number in $[-1, +1]$, sign indicates direction, magnitude indicates strength. Easy.

For **categorical** variables this breaks. You can't compute $\bar{x}$ when $x \in \{\text{Fiber optic}, \text{DSL}, \text{No}\}$. There's no "mean" of `Fiber optic`. Even if you encode the categories as numbers ($1, 2, 3$), you've imposed an artificial **ordering** ($\text{No} < \text{DSL} < \text{Fiber}$? Why?) and an artificial **distance** (is "DSL → Fiber" the same distance as "No → DSL"?). The correlation you compute will be meaningless because it reflects your encoding, not the data.

We need a different tool. **Cramér's V is that tool.** It measures how strongly two categorical variables are associated, on a clean 0-to-1 scale that's interpretable and comparable across feature pairs.

---

## 2. The Chi-Square Test of Independence

Cramér's V is built on top of the **chi-square test of independence** ($\chi^2$), so we need that first.

### Setup

Take two categorical variables, $X$ and $Y$. Build a **contingency table** (crosstab) of their joint counts:

|             | $Y = y_1$ | $Y = y_2$ | $\cdots$ | $Y = y_c$ | Total |
|-------------|-----------|-----------|----------|-----------|-------|
| $X = x_1$   | $O_{11}$  | $O_{12}$  | $\cdots$ | $O_{1c}$  | $R_1$ |
| $X = x_2$   | $O_{21}$  | $O_{22}$  | $\cdots$ | $O_{2c}$  | $R_2$ |
| $\vdots$    | $\vdots$  | $\vdots$  | $\ddots$ | $\vdots$  | $\vdots$ |
| $X = x_r$   | $O_{r1}$  | $O_{r2}$  | $\cdots$ | $O_{rc}$  | $R_r$ |
| **Total**   | $C_1$     | $C_2$     | $\cdots$ | $C_c$     | $n$   |

where $O_{ij}$ is the **observed** count for cell $(i, j)$, $R_i$ is the $i$-th row total, $C_j$ is the $j$-th column total, and $n$ is the grand total.

### The null hypothesis

**$H_0$: $X$ and $Y$ are independent** — knowing $X$ tells you nothing about $Y$.

If $X$ and $Y$ are independent, then for each cell:

$$P(X = x_i \cap Y = y_j) = P(X = x_i) \cdot P(Y = y_j) = \frac{R_i}{n} \cdot \frac{C_j}{n}$$

So under independence, the **expected** count in cell $(i, j)$ is:

$$E_{ij} = n \cdot \frac{R_i}{n} \cdot \frac{C_j}{n} = \frac{R_i \cdot C_j}{n}$$

### The test statistic

The chi-square statistic measures the gap between observed and expected:

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Each cell contributes a "squared standardized residual." Big residuals (where observed counts are far from what independence would predict) make $\chi^2$ big. If $X$ and $Y$ are perfectly independent, $\chi^2 \to 0$.

### The test

Under $H_0$, the statistic follows a chi-square distribution with degrees of freedom:

$$\text{df} = (r - 1)(c - 1)$$

You compare $\chi^2$ to a critical value (or compute a $p$-value) to decide whether to reject independence.

### Two problems with $\chi^2$ as a measure of association strength

1. **$\chi^2$ scales with sample size.** Double the dataset (same proportions), and $\chi^2$ doubles. A weak relationship in $n = 100{,}000$ produces a larger $\chi^2$ than a strong relationship in $n = 1{,}000$. So you can't compare $\chi^2$ values across datasets of different sizes.

2. **$\chi^2$ scales with table dimensions.** A $2 \times 2$ table and a $4 \times 5$ table have different degrees of freedom and different expected scales of $\chi^2$. You can't compare them either.

These two flaws mean $\chi^2$ is great for hypothesis testing (is there a relationship?) but terrible for measuring strength (how strong is the relationship?). **Cramér's V fixes both flaws.**

---

## 3. From Chi-Square to Cramér's V

The fix is to **normalize $\chi^2$ to a $[0, 1]$ scale** that strips out the dependence on $n$ and on table dimensions.

### The formula

$$V = \sqrt{\frac{\chi^2 / n}{\min(r - 1, \, c - 1)}}$$

### Why this normalization works

There are two things to remove:

**1. Removing the sample-size dependence.** Divide $\chi^2$ by $n$:

$$\phi^2 = \frac{\chi^2}{n}$$

This quantity is called the **mean-square contingency coefficient** (a.k.a. $\phi^2$). It's $\chi^2$ per observation — scale-free with respect to $n$. For a $2 \times 2$ table, $\phi^2$ already gives a clean number in $[0, 1]$, and its square root $\phi$ is sometimes used as an association measure for $2 \times 2$ tables specifically.

**2. Removing the table-dimension dependence.** For larger tables, $\phi^2$ can exceed 1. The maximum value of $\phi^2$ is $\min(r-1, c-1)$ (this is a theorem from the linear algebra of contingency tables — the rank of the residual matrix is at most $\min(r-1, c-1)$). So we divide by that maximum:

$$V^2 = \frac{\phi^2}{\min(r-1, \, c-1)} = \frac{\chi^2}{n \cdot \min(r-1, \, c-1)}$$

Then take the square root to get the association measure on the same scale as a correlation:

$$\boxed{V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, \, c-1)}}}$$

### Properties

- $V \in [0, 1]$ — always non-negative; no direction information.
- $V = 0$ ⟺ $X$ and $Y$ are perfectly independent.
- $V = 1$ ⟺ knowing $X$ determines $Y$ exactly (or vice versa).
- $V$ is **symmetric**: $V(X, Y) = V(Y, X)$.
- $V$ is **scale-free** — comparable across feature pairs, sample sizes, and table dimensions.

### Worked micro-example

Suppose we have a $2 \times 2$ table from $n = 200$ customers:

|              | Churn = No | Churn = Yes | Total |
|--------------|-----------|-------------|-------|
| Contract = M2M | 40 | 60 | 100 |
| Contract = Annual | 90 | 10 | 100 |
| **Total** | 130 | 70 | 200 |

Expected counts under independence:
- $E_{11} = 100 \cdot 130 / 200 = 65$
- $E_{12} = 100 \cdot 70 / 200 = 35$
- $E_{21} = 100 \cdot 130 / 200 = 65$
- $E_{22} = 100 \cdot 70 / 200 = 35$

$$\chi^2 = \frac{(40-65)^2}{65} + \frac{(60-35)^2}{35} + \frac{(90-65)^2}{65} + \frac{(10-35)^2}{35} = 9.62 + 17.86 + 9.62 + 17.86 = 54.95$$

With $r = c = 2$, $\min(r-1, c-1) = 1$:

$$V = \sqrt{\frac{54.95}{200 \cdot 1}} = \sqrt{0.275} \approx 0.52$$

A strong association — about what we'd expect when contract type drives churn so visibly.

---

## 4. Interpretation Scale

Cramér's V doesn't come with a built-in cut-off. The conventional guideline (originally from Cohen, 1988, for effect sizes in social-science research):

| Cramér's V | Interpretation |
|---|---|
| $0.00 - 0.05$ | Negligible — likely noise |
| $0.05 - 0.10$ | Weak association |
| $0.10 - 0.20$ | Moderate-weak association |
| $0.20 - 0.40$ | Moderate-strong association |
| $0.40 - 0.60$ | Strong association |
| $0.60 - 0.80$ | Very strong association |
| $0.80 - 1.00$ | Near-deterministic relationship |

**These are guidelines, not laws.** Real-world ML problems rarely produce $V > 0.5$ between a single feature and a binary target — even strongly predictive features usually land in the $0.2$–$0.4$ range. If you ever compute $V > 0.95$ between a feature and your target, **suspect label leakage** — your feature probably encodes the answer.

### Telco context

For Project 01, the top features ranked by Cramér's V against `Churn`:

| Rank | Feature | $V$ | Category |
|---|---|---|---|
| 1 | Contract | 0.41 | Strong |
| 2 | OnlineSecurity | 0.35 | Moderate-strong |
| 3 | TechSupport | 0.34 | Moderate-strong |
| 4 | InternetService | 0.32 | Moderate-strong |
| 5 | PaymentMethod | 0.30 | Moderate-strong |

These are realistic values for a real-world churn problem. No single feature is dominant; the predictive power will come from combining several moderate signals.

---

## 5. Bergsma's Bias Correction

Cramér's V has a known small-sample bias: it tends to overestimate association in small datasets. **Bergsma (2013)** proposed a bias correction:

$$\tilde{V} = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r} - 1, \, \tilde{c} - 1)}}$$

where:

$$\tilde{\phi}^2 = \max\!\left(0, \; \phi^2 - \frac{(r - 1)(c - 1)}{n - 1}\right)$$

$$\tilde{r} = r - \frac{(r - 1)^2}{n - 1}, \qquad \tilde{c} = c - \frac{(c - 1)^2}{n - 1}$$

### Does it matter in practice?

For datasets where $n$ is large relative to table dimensions (say, $n > 50 \cdot r \cdot c$), the correction is negligible. For the Telco dataset ($n = 7{,}043$, largest table $4 \times 2 = 8$ cells), the correction changes Cramér's V by approximately $\sim 10^{-4}$ — invisible.

For small samples ($n < 200$), or tables with many cells where some are sparse, the correction matters. Always worth mentioning in a methodological note even if you don't apply it, so reviewers know you're aware.

The `scipy.stats.contingency.association()` function in modern scipy supports the corrected version via `method='cramer'` and a `correction=True` flag.

---

## 6. Limitations and Caveats

### Cramér's V is direction-agnostic

$V(X, Y) = V(Y, X)$. It tells you "$X$ and $Y$ are associated" but not "which category of $X$ drives which outcome of $Y$." For direction, you need:
- The contingency table itself (proportions per row/column),
- A stacked-bar visualization,
- Or Theil's U (an asymmetric variant — see §8 below).

### Cramér's V doesn't capture conditional / interaction effects

$V(\text{Contract}, \text{Churn})$ measures only the marginal association. If `Contract` only matters for fiber-optic customers and is irrelevant for DSL customers, Cramér's V gives you the average effect, not the conditional one. Interaction effects show up in modeling (tree splits, interaction terms), not in pairwise association measures.

### Cramér's V conflates many small effects with one large effect

A feature with 10 categories where each contributes a small effect can produce the same $V$ as a feature with 2 categories where the difference is dramatic. The $V$ is a summary statistic — drilling into the contingency table reveals what's actually going on.

### Sparse cells inflate the test statistic

If any expected cell count $E_{ij} < 5$, the chi-square approximation becomes unreliable (the $\chi^2$ distribution is an asymptotic approximation to the multinomial). For sparse tables, use **Fisher's exact test** instead of chi-square, or merge small categories first. The "expected count $\geq 5$" rule is rough — modern advice is "no more than 20% of cells with $E < 5$, and no $E < 1$."

### Cramér's V can be gamed by category cardinality

A feature with 100 unique values will mechanically produce higher chi-square (more cells, more chances for residuals) than a feature with 2 values, even if the true association is weaker. The $\min(r-1, c-1)$ normalization helps but doesn't fully eliminate this. **Bergsma's correction partially addresses this; mutual information handles it better for very high cardinality features.**

### Cramér's V doesn't handle ordinal information

If your category has natural ordering (`Low < Medium < High`), Cramér's V ignores it. For ordinal-ordinal associations, use **Kendall's tau** or **Spearman's rho**. For ordinal-categorical, **Goodman-Kruskal's gamma** or the **Mann-Whitney U** related measures.

---

## 7. Python Implementation

### From scratch (the version you'll want to understand)

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x, y):
    """
    Cramér's V — measure of association between two categorical variables.
    
    Parameters
    ----------
    x, y : array-like of categorical values (same length)
    
    Returns
    -------
    float in [0, 1] : 0 = independence, 1 = perfect association
    """
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    r, c = confusion.shape
    denominator = min(r - 1, c - 1)
    if denominator == 0:
        return 0.0
    return np.sqrt((chi2 / n) / denominator)
```

### With Bergsma's bias correction

```python
def cramers_v_corrected(x, y):
    """Bias-corrected Cramér's V (Bergsma 2013)."""
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    r, c = confusion.shape
    
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corrected = r - (r - 1)**2 / (n - 1)
    c_corrected = c - (c - 1)**2 / (n - 1)
    denominator = min(r_corrected - 1, c_corrected - 1)
    if denominator <= 0:
        return 0.0
    return np.sqrt(phi2_corrected / denominator)
```

### Using scipy directly (modern scipy ≥ 1.7)

```python
from scipy.stats.contingency import association

V = association(pd.crosstab(x, y), method='cramer')
# correction=True for Bergsma's correction (newer scipy)
```

### Ranking all categorical features against a target

```python
def cramers_v_ranking(df, target, categorical_cols):
    """Rank all categorical features by association with target."""
    results = []
    for col in categorical_cols:
        v = cramers_v(df[col], df[target])
        results.append({'feature': col, 'cramers_v': round(v, 4)})
    return (pd.DataFrame(results)
            .sort_values('cramers_v', ascending=False)
            .reset_index(drop=True))
```

---

## 8. The Broader Landscape — Association Measures Across Variable Types

Cramér's V handles **categorical × categorical**. For other combinations, different tools apply. The full landscape:

### Numerical × Numerical

| Tool | When to use | Range | Notes |
|---|---|---|---|
| **Pearson $r$** | Linear relationships, roughly normal data | $[-1, +1]$ | Most common; fails for non-linear or skewed data |
| **Spearman $\rho$** | Monotonic (not necessarily linear) relationships | $[-1, +1]$ | Rank-based; robust to outliers and non-linearity |
| **Kendall $\tau$** | Monotonic relationships, small samples | $[-1, +1]$ | Less power than Spearman but more interpretable as a probability |
| **Distance correlation** | Any dependence (linear or non-linear) | $[0, 1]$ | $= 0$ iff variables are independent (true independence test) |

### Numerical × Binary Categorical

| Tool | When to use | Notes |
|---|---|---|
| **Point-biserial correlation** | Direct correlation between continuous and binary | Mathematically equivalent to Pearson with binary encoded as 0/1 |
| **Independent $t$-test / Mann-Whitney U** | Hypothesis testing only | Gives $p$-value, not strength |
| **Cohen's $d$** | Standardized effect size | $d = (\mu_1 - \mu_2) / s_{\text{pooled}}$; interpretable |
| **AUC of single feature** | Treats feature as a "classifier" | Equivalent to Mann-Whitney U, in $[0, 1]$ |

### Numerical × Multi-class Categorical

| Tool | When to use | Range | Notes |
|---|---|---|---|
| **ANOVA $F$-statistic** | Compare means across groups | $[0, \infty)$ | Scale-dependent; like $\chi^2$ for numericals |
| **$\eta^2$ (eta squared)** | Effect size from ANOVA | $[0, 1]$ | Proportion of variance explained; analogous to Cramér's V |
| **$\omega^2$ (omega squared)** | Bias-corrected effect size | $[0, 1]$ | Better than $\eta^2$ for small samples |
| **Kruskal-Wallis $H$** | Non-parametric alternative to ANOVA | — | Use when group distributions aren't normal |

### Categorical × Categorical

| Tool | When to use | Range | Notes |
|---|---|---|---|
| **Cramér's V** | General-purpose; symmetric | $[0, 1]$ | The default choice |
| **Phi coefficient ($\phi$)** | Specifically $2 \times 2$ tables | $[-1, +1]$ | Equivalent to Pearson on 0/1 encoding |
| **Theil's U** (uncertainty coefficient) | When direction matters | $[0, 1]$ | Asymmetric: $U(X \mid Y) \ne U(Y \mid X)$ |
| **Goodman-Kruskal $\lambda$** | Predictive association | $[0, 1]$ | "How much does knowing $X$ reduce error in predicting $Y$?" |
| **Goodman-Kruskal $\gamma$** | Ordinal categorical variables | $[-1, +1]$ | Like rank correlation for ordinals |
| **Fisher's exact test** | Sparse $2 \times 2$ tables | — | Exact $p$-value, no chi-square approximation |

### Mixed Types (Numerical + Categorical + Ordinal)

| Tool | When to use | Range | Notes |
|---|---|---|---|
| **Phik ($\phi_K$)** | Unified across all variable types | $[0, 1]$ | Re-bins numerical features into categories; handles mixed types in one matrix; Q15 of Project 01 |
| **Mutual Information (MI)** | Any dependence, any types | $[0, \infty)$ | From information theory; sklearn's `mutual_info_classif` is widely used for feature selection |
| **Normalized MI (NMI)** | Bounded version of MI | $[0, 1]$ | Useful for clustering-quality assessment |
| **Maximal Information Coefficient (MIC)** | Any non-linear relationship | $[0, 1]$ | Powerful but slow; controversial in some statistical communities |

### Theil's U — the asymmetric alternative

Worth a closer look because it solves Cramér's V's direction-agnostic limitation.

$$U(Y \mid X) = \frac{H(Y) - H(Y \mid X)}{H(Y)}$$

where $H(Y)$ is the Shannon entropy of $Y$ and $H(Y \mid X)$ is the conditional entropy. Interpretation: **"How much uncertainty about $Y$ is reduced by knowing $X$, as a fraction of $Y$'s total uncertainty?"**

Because $U$ is asymmetric, $U(\text{Churn} \mid \text{Contract})$ tells you "how much does knowing Contract reduce churn uncertainty" — different from $U(\text{Contract} \mid \text{Churn})$ which tells you "how much does knowing churn reduce contract uncertainty." For feature ranking, $U(\text{target} \mid \text{feature})$ is what you usually want.

```python
from sklearn.metrics import mutual_info_score
import math

def theils_u(x, y):
    """Theil's U(y|x): uncertainty in y explained by knowing x."""
    mi = mutual_info_score(x, y)
    _, counts_y = np.unique(y, return_counts=True)
    p_y = counts_y / counts_y.sum()
    h_y = -np.sum(p_y * np.log(p_y))
    if h_y == 0:
        return 0.0
    return mi / h_y
```

---

## 9. Choosing the Right Tool — Decision Guide

```
Two variables. What types are they?
│
├── Both numerical?
│   ├── Linear relationship?       → Pearson r
│   ├── Monotonic (non-linear)?    → Spearman ρ
│   └── Any dependence?            → Distance correlation
│
├── One numerical, one binary?
│   ├── Want strength?             → Point-biserial r / Cohen's d
│   └── Want p-value?              → t-test / Mann-Whitney U
│
├── One numerical, one multi-class categorical?
│   ├── Effect size?               → η² or ω²
│   └── Significance?              → ANOVA F-test or Kruskal-Wallis
│
├── Both categorical?
│   ├── Symmetric strength?        → Cramér's V
│   ├── Want direction?            → Theil's U (asymmetric)
│   ├── Both 2-level?              → Phi coefficient
│   ├── Sparse cells?              → Fisher's exact test
│   └── Ordinal?                   → Goodman-Kruskal γ
│
└── Mixed / any types?
    ├── Unified matrix?            → Phik (φK)
    └── Any dependence?            → Mutual Information
```

### Quick rule for ML feature ranking

If you're doing exploratory feature ranking for a classification or regression problem and don't want to think about all this:

- **For classification (categorical target):** use `sklearn.feature_selection.mutual_info_classif`. It handles mixed feature types automatically.
- **For regression (numerical target):** use `mutual_info_regression`.
- **For interpretability / a single matrix:** use `phik`.

For Project 01 specifically (Q13 → Q15), the progression is:
1. Cramér's V for categorical × Churn (Q13)
2. Pearson correlation matrix for numerical × numerical (Q14)
3. Phik for all-features × Churn in one unified ranking (Q15)

---

## 10. Practical Workflow for Feature Ranking

For every supervised ML project, run this workflow during EDA:

1. **Identify variable types.** Numerical, categorical, ordinal. Cast `SeniorCitizen`-style int-but-actually-categorical features explicitly.

2. **Univariate associations against the target.** For each feature, compute the appropriate single-feature association measure (Cramér's V for categorical, point-biserial / Cohen's d / AUC for numerical). Rank.

3. **Form expectations.** Write down which features should rank high, which should rank low, based on domain knowledge. **Then compare to the data.**

4. **Investigate surprises.** Features that rank higher or lower than expected are the most interesting EDA findings. They often signal data leakage, hidden confounds, or genuinely counterintuitive business effects.

5. **Cross-check with a unified method.** Run phik or mutual information across all feature types. If your Cramér's V ranking and your phik ranking disagree wildly, investigate why.

6. **Don't drop features based on univariate ranking alone.** A feature that's weak alone may be strong in combination (interaction effects). Univariate ranking is a *guide*, not a *decision*. Keep weak features through to modeling; let regularization or feature importance from the trained model decide what to drop.

7. **Watch for near-ties between similar features.** OnlineSecurity ($V = 0.347$) and TechSupport ($V = 0.343$) being nearly tied in the Telco dataset is a multicollinearity signal — these two features carry overlapping information. Either combine them or accept that their individual importances in a trained model will be split between them.

8. **Document the rankings.** Save the ranked tables and visualizations. You'll reference them when interpreting model feature importance later — comparing "what EDA said matters" vs "what the model learned matters" is a great sanity check.

---

## Appendix — Cheat sheet

| Variable types | Strength measure (range) | Direction? | Hypothesis test |
|---|---|---|---|
| Num × Num | Pearson $r$ $[-1, 1]$ | yes | $t$-test |
| Num × Num (rank) | Spearman $\rho$ $[-1, 1]$ | yes | Spearman test |
| Num × Binary | Point-biserial $[-1, 1]$ / Cohen's $d$ | yes | $t$-test |
| Num × Multi-cat | $\eta^2$ $[0, 1]$ | no | ANOVA $F$ |
| Cat × Cat | **Cramér's V $[0, 1]$** | no | $\chi^2$ |
| Cat × Cat (asymmetric) | Theil's U $[0, 1]$ | yes | LR test |
| 2 × 2 cat | $\phi$ $[-1, 1]$ | yes | $\chi^2$ or Fisher |
| Any × Any | Phik $\phi_K$ $[0, 1]$ | no | bootstrap |
| Any × Any | Mutual Info $[0, \infty)$ | no | permutation |

---

## Key references for further reading

- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press. — The original derivation.
- Bergsma, W. (2013). "A bias-correction for Cramér's V and Tschuprow's T." *Journal of the Korean Statistical Society*. — The bias correction.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. — The effect-size guidelines used widely in social sciences.
- Baak, M. et al. (2020). "A new correlation coefficient between categorical, ordinal and interval variables with Pearson characteristics." *Computational Statistics & Data Analysis*. — The phik paper.

---

*Notes by Rahul Das — Project 01 (Telco Customer Churn), Data Science Portfolio.*
