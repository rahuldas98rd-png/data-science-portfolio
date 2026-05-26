# Correlation and Multicollinearity — Notes & Reference

> **Personal study notes built during Project 01 (Telco Customer Churn).**
> Reusable for every regression and linear classification project.

---

## Table of Contents

1. [Why This Topic Matters](#1-why-this-topic-matters)
2. [Pearson Correlation — The Foundation](#2-pearson-correlation--the-foundation)
3. [Spearman, Kendall, and Rank-Based Correlations](#3-spearman-kendall-and-rank-based-correlations)
4. [Partial and Semi-Partial Correlation](#4-partial-and-semi-partial-correlation)
5. [Anscombe's Quartet — Why Correlation Lies](#5-anscombes-quartet--why-correlation-lies)
6. [Correlation vs Causation](#6-correlation-vs-causation)
7. [Multicollinearity — What It Is and Why It Matters](#7-multicollinearity--what-it-is-and-why-it-matters)
8. [The Mathematical Source of the Problem](#8-the-mathematical-source-of-the-problem)
9. [Detection — VIF, Condition Number, and Beyond](#9-detection--vif-condition-number-and-beyond)
10. [Treatment Strategies](#10-treatment-strategies)
11. [Why Tree-Based Models Don't Care (Mostly)](#11-why-tree-based-models-dont-care-mostly)
12. [Practical Workflow](#12-practical-workflow)
13. [Cheat Sheet](#13-cheat-sheet)

---

## 1. Why This Topic Matters

Two distinct but related questions live in this topic:

1. **How strongly are two variables related?** Answered by correlation coefficients.
2. **What happens when my predictor features are related to each other?** Answered by multicollinearity diagnostics.

Most analysts use Pearson correlation reflexively and move on. That's a mistake. Correlation has well-known failure modes (non-linearity, outlier sensitivity, distributional assumptions), and **multicollinearity silently destroys the interpretability of every linear model you'll ever fit** if you don't check for it. The cost of ignoring this is models whose coefficients flip sign between random seeds — accurate on paper, useless in practice.

---

## 2. Pearson Correlation — The Foundation

### Definition

For two numerical variables $X$ and $Y$ with $n$ paired observations:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Equivalently, in terms of covariance and standard deviations:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

Pearson $r$ is **covariance normalized by the product of standard deviations** — it strips the scale out so the answer is comparable across variables in different units.

### Properties

- $r \in [-1, +1]$
- $r = +1$ ⟺ perfect positive linear relationship
- $r = -1$ ⟺ perfect negative linear relationship
- $r = 0$ ⟺ no linear relationship (but possibly strong non-linear relationship)
- **Symmetric:** $r(X, Y) = r(Y, X)$
- **Scale-invariant:** $r(aX + b, cY + d) = r(X, Y) \cdot \text{sign}(ac)$
- **Unit-free:** correlation between height (cm) and weight (kg) is the same as between height (inches) and weight (lbs)

### Interpretation conventions

| $|r|$ | Conventional label |
|---|---|
| $0.00 - 0.10$ | Negligible |
| $0.10 - 0.30$ | Weak |
| $0.30 - 0.50$ | Moderate |
| $0.50 - 0.70$ | Strong |
| $0.70 - 0.90$ | Very strong |
| $0.90 - 1.00$ | Near-deterministic |

**These are conventions, not laws.** In some fields (e.g., social sciences) $r = 0.3$ is considered impressive; in others (e.g., physics) $r = 0.95$ might be considered modest. Always state the context.

### The $r^2$ interpretation

Squaring the correlation gives the **coefficient of determination**:

$$r^2 = \text{fraction of variance in } Y \text{ explained by linear regression on } X$$

So $r = 0.83$ means linear regression of $Y$ on $X$ explains $0.83^2 \approx 69\%$ of the variance in $Y$. This is often more meaningful than $r$ itself for communication.

### Assumptions for inference

Pearson $r$ as a point estimate has no assumptions. But if you want to do hypothesis testing or build confidence intervals on $r$:

1. **Linearity** — the relationship between $X$ and $Y$ is linear.
2. **Bivariate normality** — $(X, Y)$ jointly follow a multivariate normal distribution.
3. **Homoscedasticity** — variance of $Y$ doesn't change with $X$.
4. **No influential outliers** — single points shouldn't drive the result.

Violations of these don't invalidate the calculation, but they make the $p$-value and confidence interval untrustworthy.

### When Pearson fails

- **Non-linear relationships:** $Y = X^2$ has $r = 0$ on symmetric data despite a perfect deterministic relationship.
- **Outliers:** a single extreme point can swing $r$ from 0.3 to 0.9 or vice versa.
- **Truncated data:** if you've restricted the range of $X$, you'll underestimate $r$.
- **Mixed populations:** two clusters with no within-cluster correlation can produce high $r$ from the between-cluster variation (Simpson's paradox).
- **Ordinal / non-interval data:** the math assumes interval scale; treating Likert ratings (1–5) as Pearson-compatible is technically wrong but widely done.

For most of these failures, **Spearman is the answer.**

---

## 3. Spearman, Kendall, and Rank-Based Correlations

### Spearman's $\rho$

Compute Pearson correlation on the **ranks** of the data rather than the raw values:

$$\rho = \frac{\sum_{i}(R_i^X - \overline{R^X})(R_i^Y - \overline{R^Y})}{\sqrt{\sum_i(R_i^X - \overline{R^X})^2 \cdot \sum_i(R_i^Y - \overline{R^Y})^2}}$$

where $R_i^X$ is the rank of $x_i$ among all $x$'s.

If there are no ties, this simplifies to:

$$\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}, \quad d_i = R_i^X - R_i^Y$$

**What Spearman tests:** monotonic relationships, not strictly linear. If $Y$ grows with $X$ in any consistent direction (even non-linearly), Spearman picks it up.

**When to use:**
- Outliers present
- Non-linear but monotonic relationship
- Ordinal data
- Non-normal distributions
- Small samples where assumptions of Pearson are doubtful

### Kendall's $\tau$

Counts **concordant** and **discordant** pairs:

$$\tau = \frac{(\#\text{concordant pairs}) - (\#\text{discordant pairs})}{\binom{n}{2}}$$

A pair $(i, j)$ is concordant if both $x_i < x_j$ and $y_i < y_j$ (or both reversed). Discordant if one comparison goes one way and the other goes the opposite.

**Properties vs Spearman:**
- Generally smaller magnitudes than $\rho$ for the same data
- More interpretable as a probability: $\tau$ ≈ $2 \cdot P(\text{concordant}) - 1$
- More robust to outliers and ties
- Slower to compute ($O(n^2)$ vs Spearman's $O(n \log n)$)
- Preferred for small samples ($n < 30$) where it has better statistical properties

### Comparison summary

| Method | What it measures | Range | When to use |
|---|---|---|---|
| Pearson $r$ | Linear relationship | $[-1, +1]$ | Continuous, roughly normal, no outliers |
| Spearman $\rho$ | Monotonic relationship | $[-1, +1]$ | Outliers, non-linear monotonic, ordinal data |
| Kendall $\tau$ | Concordance of pairs | $[-1, +1]$ | Small samples, very robust to outliers, ties |

**Rule of thumb:** If your Pearson and Spearman correlations agree closely (within ~0.05), Pearson assumptions are roughly satisfied. If they disagree substantially, **trust Spearman** — the disagreement is usually evidence of non-linearity or outliers that bias Pearson.

### Distance correlation (bonus)

A modern measure: **distance correlation** $\text{dCor}$ is zero if and only if two random variables are statistically independent (true independence test, unlike Pearson which only detects linear dependence). Range $[0, 1]$. Implemented in `scipy.stats.distance_correlation` and the `dcor` package. Use when you need to detect any form of dependence including highly non-linear ones.

---

## 4. Partial and Semi-Partial Correlation

These measure the relationship between $X$ and $Y$ **after controlling for one or more other variables** $Z$.

### Partial correlation

$$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} \cdot r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}$$

Interpretation: the correlation between $X$ and $Y$ after removing the linear effect of $Z$ from **both** $X$ and $Y$. Answers: "After accounting for $Z$, is there still a relationship?"

### Semi-partial (part) correlation

Removes $Z$'s effect from only one of the variables (usually $X$):

$$r_{Y(X \cdot Z)} = \frac{r_{XY} - r_{XZ} \cdot r_{YZ}}{\sqrt{1 - r_{XZ}^2}}$$

Used when you want to know how much $X$ uniquely contributes to predicting $Y$ beyond what $Z$ already explains.

### Why this matters for the Telco project

In Q10, you saw `tenure` and `TotalCharges` are highly correlated. If you regress `MonthlyCharges` on `Churn`, some of that relationship goes through tenure (long-tenured customers have different MonthlyCharges patterns). Partial correlation lets you ask: "Is MonthlyCharges related to Churn *independently of* tenure?" The answer is often what matters for modeling, not the raw bivariate correlation.

```python
import pingouin as pg
pg.partial_corr(data=df, x='MonthlyCharges', y='Churn_binary', covar='tenure')
```

---

## 5. Anscombe's Quartet — Why Correlation Lies

In 1973, Francis Anscombe constructed four datasets with **identical** summary statistics:

- $n = 11$ each
- Mean of $x$: 9.0, mean of $y$: 7.5
- Variance of $x$: 11.0, variance of $y$: 4.12
- **Pearson correlation: 0.816** in all four
- Regression line: $y = 3.00 + 0.500x$ in all four

But when plotted, the four datasets look **completely different**:

1. **Quartet I:** A normal-looking linear relationship with some scatter.
2. **Quartet II:** A perfect parabola — clear non-linear relationship that Pearson misses.
3. **Quartet III:** A perfectly linear relationship distorted by a single outlier.
4. **Quartet IV:** Almost all $x$ values are identical except for one influential point that drives the correlation single-handedly.

**The lesson:** correlation is a one-number summary. It cannot replace looking at the data. **Always plot your relationships before trusting a correlation coefficient.**

This is why your Q10 scatter plot was essential — it would have caught any Anscombe-type issues in the `tenure` ↔ `TotalCharges` relationship.

The modern extension is the **Datasaurus Dozen**: 13 datasets with identical summary statistics including correlation, but radically different visual shapes (one of them literally looks like a dinosaur). Search "Datasaurus" to see the canonical example.

---

## 6. Correlation vs Causation

The cliché is "correlation does not imply causation," and it's true. But the more useful framing is: **correlation can arise from at least four mechanisms**, and the analyst's job is to figure out which one applies.

When $X$ and $Y$ are correlated, possibilities include:

1. **$X$ causes $Y$** — what we usually assume.
2. **$Y$ causes $X$** — reverse causation. Spend more on marketing → sales go up. But also: sales go up → company has more budget → marketing budget goes up.
3. **Common cause $Z$** — a third variable drives both. Ice cream sales correlate with drowning deaths because summer (hot weather) causes both.
4. **Selection / sampling bias** — your data collection process creates correlations that don't exist in the population. Hospital patients with both disease A and disease B may be correlated even if they're independent in the general population.
5. **Pure coincidence** — with enough variables, some correlations will be spurious. See *Tyler Vigen's Spurious Correlations* for entertaining examples (US cheese consumption correlates 0.95 with bedsheet-tangling deaths).

**Why this matters for Telco:** In Q12 you found that `PaperlessBilling` customers churn more. You correctly noted that PaperlessBilling probably doesn't *cause* churn — it's a proxy for tech-savvy customers who also happen to shop around for alternatives. The actual causal driver is "customer type"; PaperlessBilling and churn behavior are both downstream effects. Acting on the wrong causal model (e.g., forcing customers to receive paper bills) wouldn't fix anything.

Distinguishing correlation from causation requires either:
- A **controlled experiment** (A/B test, randomized trial)
- **Causal inference methods** (instrumental variables, regression discontinuity, propensity scores, DAGs)

Neither is part of standard EDA, but knowing the distinction is what separates analysts from spreadsheet-runners.

---

## 7. Multicollinearity — What It Is and Why It Matters

### Definition

**Multicollinearity** = high linear dependence among predictor features in a regression model. Note carefully: it's about predictors being correlated **with each other**, not with the target.

Two flavors:

- **Perfect multicollinearity:** one feature is an exact linear combination of others. Example: `total_revenue = price × quantity` — if both `price`, `quantity`, AND `total_revenue` are in the model, the design matrix is rank-deficient and the model literally cannot be fit.
- **Near multicollinearity:** features are highly but not perfectly correlated. The model can be fit, but the coefficients are unstable.

In practice, "multicollinearity" without qualifier means *near* multicollinearity — perfect cases get caught by sklearn errors immediately.

### Why it's specifically a *linear-model* problem

A linear regression fits:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k$$

The interpretation of $\beta_j$ is: **"the expected change in $y$ when $x_j$ increases by one unit, holding all other features constant."**

If $x_1$ and $x_2$ are nearly identical (high correlation), there's no real-world data where $x_1$ changes while $x_2$ stays fixed. The model is being asked an unanswerable question. The mathematical machinery still produces *some* answer — but it'll be wildly unstable.

### The symptoms of multicollinearity

| Symptom | What you observe |
|---|---|
| Sign flips | Same model, different train/test split, opposite sign on a coefficient |
| Inflated standard errors | Coefficients have huge confidence intervals, individual $p$-values are large |
| High R² with no significant features | Model fits well overall but every individual coefficient is "not significant" |
| Sensitivity to data perturbation | Dropping a single row changes coefficients drastically |
| Counterintuitive coefficient magnitudes | Coefficient values that don't match domain intuition (e.g., negative when you expected positive) |

If you see any of these, multicollinearity is a prime suspect.

### Why predictive accuracy is usually unaffected

Crucially: **multicollinearity does not (necessarily) hurt prediction.** Two correlated features carry roughly the same information, so the model trades off between them to produce the same $\hat{y}$. Predictive accuracy on new data is fine.

What's hurt is **interpretation, stability, and inference.** If your goal is "predict who will churn," moderate multicollinearity is annoying but tolerable. If your goal is "explain *why* customers churn so the business can act," multicollinearity is fatal.

This is why "multicollinearity matters" is a more nuanced statement than "always remove correlated features." The answer depends on whether you care about coefficients (interpretation) or just predictions.

---

## 8. The Mathematical Source of the Problem

For the curious / interview-prep crowd. You can skip this section if you only need the practical results.

The OLS coefficient estimates are:

$$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$$

where $X$ is the $n \times k$ design matrix. The variance of the estimates is:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (X^T X)^{-1}$$

When features are highly correlated, $X^T X$ becomes **near-singular** (close to non-invertible). Its inverse $(X^T X)^{-1}$ has very large entries. The diagonal entries of $(X^T X)^{-1}$ are the variances of the individual coefficient estimates — so large diagonals mean large standard errors.

In the limit of perfect multicollinearity, $X^T X$ is exactly singular and $(X^T X)^{-1}$ doesn't exist. Software refuses to fit the model.

**Geometric interpretation:** The OLS solution finds the unique linear combination of features that best matches $y$. When features are nearly parallel (high correlation), many linear combinations produce nearly the same $\hat{y}$ — the "best" combination is poorly defined and sensitive to small data changes.

**Information-theoretic interpretation:** Correlated features carry overlapping information. The model can't isolate the unique contribution of each, so it splits credit arbitrarily.

---

## 9. Detection — VIF, Condition Number, and Beyond

### Method 1: Pairwise correlation matrix

The cheap first pass. Compute Pearson correlations among all predictors; flag pairs with $|r| > 0.7$ (rough rule of thumb).

**Limitation:** Only catches **two-way** multicollinearity. Three or more features can together produce a near-linear dependence that no pairwise check finds.

### Method 2: Variance Inflation Factor (VIF)

For each predictor $x_j$, regress it on **all other predictors**:

$$x_j = \alpha_0 + \alpha_1 x_1 + \cdots + \alpha_{j-1} x_{j-1} + \alpha_{j+1} x_{j+1} + \cdots + \alpha_k x_k + \varepsilon$$

Get $R_j^2$ from this regression. Then:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

**Interpretation:** $\text{VIF}_j$ tells you how much the variance of $\hat{\beta}_j$ is inflated due to its correlation with other predictors. A VIF of 5 means $\hat{\beta}_j$ has 5× the variance it would have if $x_j$ were uncorrelated with everything else.

**Cut-offs (rough conventions):**

| VIF | Interpretation |
|---|---|
| $1 - 2$ | No multicollinearity |
| $2 - 5$ | Mild — usually fine |
| $5 - 10$ | Moderate — investigate |
| $> 10$ | Serious — take action |

Some textbooks use 4 instead of 5; others use 30 as the cutoff for "serious." The conventions vary. Use 5 as default and 10 as the alarm threshold.

**The advantage over pairwise correlation:** VIF catches multi-way dependencies. If $x_1 = x_2 + x_3$ with no high pairwise correlations, VIF reveals the problem; pairwise correlation doesn't.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def compute_vif(df, features):
    X = sm.add_constant(df[features])  # important: include the intercept
    return pd.DataFrame({
        'feature': features,
        'VIF': [variance_inflation_factor(X.values, i+1) for i in range(len(features))]
    }).sort_values('VIF', ascending=False)
```

### Method 3: Condition number

Compute the **condition number** of $X^T X$ (or equivalently, of $X$ itself):

$$\kappa(X) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues of $X^T X$. A condition number > 30 indicates moderate multicollinearity; > 100 indicates serious problems.

```python
import numpy as np
condition_number = np.linalg.cond(X.values)
```

Useful when VIF gives ambiguous results, or when you want a single global multicollinearity diagnostic.

### Method 4: Eigenvalue analysis

If $X^T X$ has eigenvalues with a wide spread (largest >> smallest), multicollinearity is present. The smallest eigenvalue identifies the direction of near-linear dependence; the corresponding eigenvector tells you which feature combination is problematic.

### What to do with categorical features

The methods above are designed for continuous features. For categorical features after one-hot encoding:

- **Dummy variable trap:** If you one-hot encode a categorical feature with $k$ categories into $k$ binary columns, the columns are perfectly multicollinear (their sum is always 1). Solution: use `drop_first=True` in `pd.get_dummies()` to leave one category as the baseline.
- **VIF on dummies:** Computing VIF on one-hot-encoded features is technically valid but interpretations get tricky. Better to check Cramér's V or phik *between* categorical features first.

---

## 10. Treatment Strategies

In rough order of severity, from "just delete the problem" to "engineer your way around it":

### Strategy 1: Drop one of the correlated features

The simplest fix. If $x_1$ and $x_2$ have $r > 0.9$, they carry essentially the same information. Drop one.

**How to choose which to drop:**
- The one with lower correlation to the target (less predictive value)
- The one that's harder to interpret to a business stakeholder
- The one that's more derived / less primary (e.g., drop `TotalCharges` which is derived from `tenure × MonthlyCharges`)
- The one with more missing values or quality issues

**Risk:** If the dropped feature has any unique signal, you lose it. Usually fine when correlations are very high; risky when correlations are moderate (0.7–0.85).

### Strategy 2: Combine features into a derived feature

Replace the correlated set with a single feature that captures the joint information without redundancy.

**Examples:**
- `TotalCharges` + `tenure` → `avg_monthly_spend = TotalCharges / max(tenure, 1)`
- `height` + `weight` → BMI
- `revenue` + `cost` → `profit_margin`

The derived feature is uncorrelated with the originals (if engineered correctly) and often more meaningful.

**Risk:** Loss of granularity. You're collapsing two signals into one.

### Strategy 3: Regularization

Use **Ridge regression (L2)** or **Elastic Net** instead of OLS. The regularization penalty shrinks correlated coefficients toward each other rather than letting them blow up to opposite extremes.

Ridge specifically: minimize

$$\sum_i (y_i - \hat{y}_i)^2 + \lambda \sum_j \beta_j^2$$

The L2 penalty $\lambda \sum \beta_j^2$ keeps coefficients small and stable. For correlated features, Ridge essentially "averages" the credit between them — neither blows up.

**Lasso (L1)** behaves differently with correlated features: it tends to pick one feature from each correlated group and zero out the rest. This can be useful (automatic feature selection) but unstable across runs.

**Elastic Net** combines both ($\alpha L_1 + (1-\alpha) L_2$) and is often the safest choice in the presence of multicollinearity.

```python
from sklearn.linear_model import Ridge, ElasticNet
model = Ridge(alpha=1.0).fit(X, y)
# or
model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y)
```

**Risk:** Coefficients are biased (shrunk toward zero). Good for prediction; the bias means coefficients no longer have clean OLS interpretation.

### Strategy 4: Principal Component Analysis (PCA)

Transform the correlated features into uncorrelated principal components, use the components as inputs.

**Pros:** Components are guaranteed uncorrelated. Reduces dimensionality.

**Cons:**
- Components are hard to interpret (linear combinations of original features)
- You lose direct connection to business meaning
- Requires careful scaling first

**When it's appropriate:** Many correlated features (10+), prediction-focused use case, interpretability not critical.

### Strategy 5: Partial Least Squares (PLS)

Like PCA but supervised — finds components that maximize correlation with the *target*, not just capture variance in features. Often beats PCA for predictive modeling with correlated features.

### Decision framework

| Situation | Recommended treatment |
|---|---|
| 2 features, $r > 0.95$, one is clearly derived | Drop the derived one |
| 2 features, $r > 0.85$, both meaningful | Combine into a derived feature, OR keep both with Ridge |
| 3+ correlated features, want interpretation | Combine into a single domain-meaningful feature |
| Many correlated features, prediction only | PCA, Ridge, or Elastic Net |
| Don't know what to do | Try Elastic Net — it handles most cases gracefully |

---

## 11. Why Tree-Based Models Don't Care (Mostly)

Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), and decision trees are **largely immune** to multicollinearity in terms of predictive performance.

**Why:** They make decisions by splitting on one feature at a time. If `tenure` and `TotalCharges` are correlated, the model might split on `tenure` at one node and on `TotalCharges` at another, but each individual split is independent of the others. There's no coefficient to flip sign — there's no coefficient at all.

The model picks whichever correlated feature gives the best split at each step, ignores the rest, and moves on. Predictive accuracy is unaffected.

### But there's a catch: feature importance gets distorted

If `tenure` and `TotalCharges` are highly correlated and both predictive, the model splits its "importance credit" between them at random — each appears moderately important rather than one being clearly dominant.

This matters when you're trying to interpret which features matter:

- **Built-in feature importance** (gini, gain, split count) understates correlated features
- **Permutation importance** has the same problem — permuting one correlated feature doesn't hurt the model much because the other one still carries the signal
- **SHAP values** *also* distribute credit between correlated features, but they do it more transparently — you can see the split

The fix: **for interpretation**, still consider de-duplicating correlated features even when using tree models. For pure prediction, leave them.

### When tree models DO have multicollinearity-like problems

- **High-cardinality categorical features:** Tree models can artificially favor features with many unique values (more splitting opportunities). This is mitigated by careful hyperparameter tuning (min_samples_leaf) or proper encoding.
- **Highly correlated rare events:** If a class is rare and several features perfectly predict it, the model can become unstable in subtle ways.

These are edge cases. For typical tabular data, tree models really don't care about correlation.

---

## 12. Practical Workflow

For every modeling project, run this sequence during EDA:

1. **Pairwise correlation matrix** for all numerical predictors. Heatmap visualization. Flag any $|r| > 0.7$.

2. **Inspect the relationships visually.** For any flagged pair, look at the scatter plot. Confirm the correlation isn't being driven by outliers or non-linearity (i.e., that Pearson is telling the truth).

3. **Cross-check with Spearman.** If Pearson and Spearman disagree substantially, the data has non-linearity or outliers. Investigate.

4. **Compute VIF** for the predictor set. Sort descending. Flag VIF > 5.

5. **Investigate features with high VIF.** Are they derivable from other features? Are they conceptually similar? What's the right action — drop, combine, or accept and regularize?

6. **Decide treatment based on modeling intent:**
   - Pure prediction with tree models → ignore multicollinearity
   - Pure prediction with linear models → use regularization (Ridge / Elastic Net)
   - Interpretation matters → drop or combine to get clean coefficients

7. **Document your decision.** In the notebook, explain why you chose your treatment strategy. "I dropped TotalCharges because it's mathematically derivable from tenure × MonthlyCharges, and removing it improved coefficient stability without hurting test AUC." This is the kind of statement that distinguishes engineering work from script-running.

8. **For sensitivity analysis:** Fit your linear model with and without the treatment. If coefficients change drastically between the two fits, multicollinearity was real and your treatment was necessary. If they don't change, the multicollinearity wasn't actually hurting interpretation.

---

## 13. Cheat Sheet

### Correlation coefficients

| Method | Range | Detects | Use when |
|---|---|---|---|
| Pearson $r$ | $[-1, +1]$ | Linear | Continuous, ~normal, no outliers |
| Spearman $\rho$ | $[-1, +1]$ | Monotonic | Outliers, non-linear monotonic, ordinal |
| Kendall $\tau$ | $[-1, +1]$ | Concordance | Small samples, ties, very robust |
| Distance correlation | $[0, 1]$ | Any dependence | True independence testing |

### Multicollinearity diagnostics

| Method | Threshold | What it catches |
|---|---|---|
| Pairwise $|r|$ | > 0.7 | Two-way correlation only |
| VIF | > 5 (warning), > 10 (alarm) | Multi-way dependencies |
| Condition number | > 30 (warning), > 100 (alarm) | Global ill-conditioning |
| Eigenvalue ratio | $\lambda_{\max}/\lambda_{\min}$ | Direction of dependence |

### Treatment strategies

| Strategy | Best for |
|---|---|
| Drop one feature | Very high correlation, one feature is clearly derived |
| Combine into derived feature | Moderate correlation, want interpretability |
| Ridge / Elastic Net | Many correlated features, prediction focus |
| PCA / PLS | Very high dimensionality with correlations |
| Ignore (use tree model) | Pure prediction, no need for coefficient interpretation |

### Model-family behavior with multicollinearity

| Model | Predictive accuracy | Coefficient interpretability | Feature importance |
|---|---|---|---|
| OLS | Unaffected | **Destroyed** | N/A |
| Ridge | Unaffected | Biased but stable | Stable |
| Lasso | Unaffected | Picks one feature per group | Unstable |
| Random Forest | Unaffected | N/A | Distorted (split credit) |
| Gradient Boosting | Unaffected | N/A | Distorted (split credit) |

### Quick decision tree

```
Do I care about coefficient interpretation?
├── No (pure prediction)
│   ├── Using linear model? → Use Ridge or Elastic Net, ignore VIF
│   └── Using tree model?   → Ignore multicollinearity entirely
│
└── Yes (need to explain "why")
    ├── Compute pairwise correlation + VIF
    ├── Any |r| > 0.85 or VIF > 10?
    │   ├── Yes → Drop or combine
    │   └── No  → Proceed, but acknowledge limitation
    └── For final model, do sensitivity analysis:
        fit with/without correlated features, check coefficient stability
```

---

## Appendix — Key formulas

$$
\begin{aligned}
\text{Pearson}    &: \quad r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}} \\[8pt]
\text{Spearman}   &: \quad \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}, \quad d_i = R_i^X - R_i^Y \\[8pt]
\text{Kendall}    &: \quad \tau = \frac{C - D}{\binom{n}{2}} \\[8pt]
\text{Partial}    &: \quad r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}} \\[8pt]
\text{VIF}_j      &= \frac{1}{1 - R_j^2}, \quad \text{where } R_j^2 \text{ is from regressing } x_j \text{ on all other features} \\[8pt]
\text{Condition number} &= \kappa(X) = \frac{\lambda_{\max}(X^T X)}{\lambda_{\min}(X^T X)}
\end{aligned}
$$

---

## Key references

- Anscombe, F. J. (1973). "Graphs in Statistical Analysis." *The American Statistician*. — The original quartet paper.
- Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. — The classic multicollinearity reference.
- Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models*, 3rd ed. — Comprehensive treatment of VIF and related diagnostics.
- O'Brien, R. M. (2007). "A Caution Regarding Rules of Thumb for Variance Inflation Factors." *Quality & Quantity*. — Discusses why fixed VIF cutoffs can mislead.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. — Accessible coverage of collinearity, Ridge, and Lasso.

---

*Notes by Rahul Das — Project 01 (Telco Customer Churn), Data Science Portfolio.*
