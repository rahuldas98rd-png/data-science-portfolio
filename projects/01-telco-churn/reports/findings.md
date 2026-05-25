# Telco Customer Churn — Findings

- **Project:** Customer churn analysis for an IBM Telco sample (7,043 subscribers, single snapshot)
- **Author:** Rahul Das
- **Date:** May 2026

---

## Executive summary

Of **7,043 customers**, **26.5% (1,869) have churned** — a class-imbalanced target where the default "predict everyone stays" baseline already scores 73.5% accuracy and is therefore useless. The single largest driver of churn is **contract type**: month-to-month subscribers churn at **15× the rate of two-year subscribers** (42.7% vs 2.8%). Behind contract, three more levers do most of the work — **payment method, internet service tier, and the presence of "stickiness" add-ons (OnlineSecurity, TechSupport)**. A small number of stacked risk factors define the high-churn customer profile, and a targeted retention program against that profile could materially shift the company's churn rate without needing a model. Modeling will then sharpen the targeting further.

---

## Top 5 findings

**1. Contract length dominates churn, by a wide margin.** Month-to-month customers churn at **42.7%**; one-year at **11.3%**; two-year at **2.8%**. Cramér's V = **0.41**, the highest of any categorical feature against Churn. The effect is partly mechanical — long-contract customers face early-termination fees — but the gap is too large to be friction alone. Contract length is also a self-selection signal: customers willing to sign two years upfront are a fundamentally stickier population. Either way, **contract is the single highest-leverage policy variable for retention.**

**2. Tenure is the strongest *numerical* signal, and the distribution is bimodal — not a smooth decay.** Churners' median tenure is **10 months**; retained customers' median is **38 months** — a 3.8× ratio with barely-overlapping IQRs. The tenure histogram has two peaks: one at 0–2 months (the danger zone) and one at 69–72 months (long-term loyalists), with a thin middle. This is **survivorship in plain sight** — customers who make it past the first year almost never leave. The implication is that **onboarding-window interventions have outsized ROI** compared to broad retention spend across the base.

**3. Electronic check is a behavioral red flag, independent of dollar amount.** Electronic-check customers churn at **45.3%** — higher than any contract type. Every other payment method (mailed check, bank transfer, credit card) sits in a tight 15–19% band. The driver isn't price; it's **friction and intent**. Auto-pay customers have set-and-forget billing and just keep paying. Electronic-check customers are actively choosing to authorize each payment — which means they're also actively choosing each month *not* to leave. **PaymentMethod is essentially a "cancel intent" sensor disguised as a billing field.**

**4. The Fiber paradox: premium internet customers churn the most.** Fiber-optic customers churn at **41.9%** vs DSL at **19.0%** and no-internet at **7.4%**. This is counterintuitive — fiber is the company's premium product and these are higher-revenue customers. The likely explanation is **expectation mismatch**: fiber customers pay a premium (median MonthlyCharges ~$90 vs ~$55 for DSL) and tolerate service issues poorly. They also tend to be on month-to-month contracts and use electronic check, so the risk factors stack. Fiber is a revenue success and a retention liability simultaneously.

**5. "Stickiness" add-ons matter; "entertainment" add-ons don't.** Customers without **OnlineSecurity** churn at 41.8% vs 14.6% with it; **TechSupport** shows almost the same gap (41.6% vs 15.2%). But **StreamingTV** and **StreamingMovies** show much weaker association (Cramér's V ~0.23 vs ~0.35 for the security/support pair). Why? Security and support create **integration friction** — switching providers means re-setting up protection and losing your support relationship. Streaming is **commoditized** — Netflix, Disney+, etc., work identically on any ISP. **The add-ons that reduce churn are the ones the customer can't easily replace.**

**Sanity check (not a finding, but worth stating):** Gender's Cramér's V against Churn is **0.0086** — essentially zero. PhoneService is also near-zero. This matches the prior domain expectation and confirms the dataset isn't behaving pathologically. If a downstream model rates gender as important, that's a bug, not a finding.

---

## High-risk customer segments

| Segment | Approx size | Churn rate | Why it's at risk |
|---|---|---|---|
| Month-to-month + Electronic check + Fiber optic | ~900 (~13%) | ~55–60% | All three top risk factors stacked |
| Month-to-month + No OnlineSecurity + No TechSupport | ~1,700 (~24%) | ~50% | No contract lock, no service stickiness |
| Tenure ≤ 6 months + Month-to-month | ~1,200 (~17%) | ~60% | Onboarding window + zero switching cost |

These three (overlapping) segments together capture the **bulk of churn volume**. Filtering to just month-to-month + electronic check already isolates a population where roughly half are about to leave.

---

## Recommendations (actionable without a model)

**1. Convert month-to-month customers to longer contracts.** Given the 15× churn ratio, even a modest conversion campaign — say, "2 months free if you commit to one year" — targeted at month-to-month customers with 6+ months of tenure is likely the highest-ROI lever the company has. This is the one number that, if moved, moves everything.

**2. Migrate electronic-check customers to auto-pay.** A behavioral campaign — small statement credit for switching to bank transfer or credit card — would attack the second-strongest signal. The mechanism isn't price; it's removing the monthly cancel-decision moment from the customer's life.

**3. Build a 0–6 month onboarding journey for new fiber customers specifically.** The fiber+new-customer segment is doubly exposed. Proactive CSM outreach at day 30 and day 90, plus a "first-month service guarantee," would target where churn is densest. Don't wait for tickets — most churners don't file any.

**4. Bundle OnlineSecurity and TechSupport free for the first 6 months of new accounts.** These add-ons cut churn by ~2.7×. Giving them away during the danger window creates the integration friction that keeps customers past month 12, after which churn collapses anyway.

---

## Open questions and next steps

- **Is the fiber-churn effect about fiber itself or about who buys fiber?** Fiber customers correlate with month-to-month + electronic check. The effect may largely disappear once those are controlled for. A multivariate model (Phase 2) is needed to separate them.
- **What does the 11-row `TotalCharges` blank represent?** All 11 are tenure=0 customers — signed up but not yet billed a full cycle. I imputed them to 0 (consistent with the `TotalCharges ≈ tenure × MonthlyCharges` identity), but their churn status is also informative: do they ever become real customers, or do brand-new signups churn pre-first-bill at unusual rates? Worth surfacing.
- **TotalCharges is mechanically redundant** with tenure × MonthlyCharges (correlation ≈ 0.83 with tenure alone). It should not be fed alongside both in any linear model — multicollinearity will distort coefficients. Tree models can absorb it.
- **PaperlessBilling churns at 33.6% vs 16.3%** — a real signal (V ≈ 0.19) but probably a *proxy* for "younger, more tech-savvy, more digitally mobile" rather than a causal lever. We can't fix churn by mailing more paper.

**Next phase:** Build a baseline logistic regression and a gradient-boosting model on the cleaned feature set. Engineer a `tenure_bucket` categorical to capture the bimodal tenure pattern that a linear term will miss. Evaluate primarily on **recall at fixed retention-team capacity** — i.e., "if the team can call 500 customers a week, which 500?" — rather than on raw AUC. The goal is operational targeting, not a leaderboard score.
