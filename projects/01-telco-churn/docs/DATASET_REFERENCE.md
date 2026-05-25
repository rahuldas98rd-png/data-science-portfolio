# Telco Customer Churn — Dataset Reference

## What is this dataset?

This is a **fictional but realistic dataset published by IBM** as a sample for their Cognos Analytics product. It simulates a **telecommunications company** (think: a US-based provider like AT&T, Verizon, or Comcast) and contains a snapshot of **7,043 customers** at a single point in time.

Each row = one customer. Each column = something the company knows about that customer.

The business problem it's built to teach is **customer churn prediction** — figuring out, from what you know about a customer today, whether they'll cancel their service in the near future.

---

## Why churn matters (the business context)

In subscription businesses (telecom, streaming, SaaS, gyms, insurance), **keeping an existing customer is 5–25× cheaper than acquiring a new one.** This is one of the most well-established findings in marketing.

A churned customer = lost revenue + wasted acquisition cost + potentially negative word-of-mouth.

So companies build churn models to answer:
1. **Who is likely to leave?** (predictive task — this is what your model does)
2. **Why are they leaving?** (interpretive task — feature importance, SHAP)
3. **What can we do about it?** (intervention — discounts, contract upgrades, support outreach)

A telecom company with millions of customers and even a 1–2% improvement in retention is worth tens of millions of dollars per year. This is why churn modeling is one of the most common real-world ML use cases.

---

## How telecom billing works (essential context)

To understand the features, you need to know how a telecom customer's account works:

- A customer **signs up** for service on some date. That's day zero.
- They pick a **contract type**: month-to-month (cancel anytime), 1-year, or 2-year (locked in, usually cheaper).
- They pick **services**: phone line, internet, and optional add-ons (streaming, security, tech support, etc.).
- Every month, they're billed based on the services they have. This is their **MonthlyCharges**.
- The **TotalCharges** is the cumulative sum of everything they've paid since signing up.
- **Tenure** is how many months they've been a customer.

This gives you a critical relationship:

```
TotalCharges ≈ tenure × MonthlyCharges
```

Not exactly equal (prices change, promotions, partial months), but close. This is why these three columns are highly correlated — and it's a clue you'll need for Q3 and later for multicollinearity discussions.

---

## Column-by-column reference

### Identifier
| Column | Type | Description | Notes |
|---|---|---|---|
| `customerID` | string | Unique customer identifier | Drop before modeling — no predictive value |

### Demographics (4 columns)
| Column | Type | Values | What it really means |
|---|---|---|---|
| `gender` | categorical | Male, Female | Customer's gender |
| `SeniorCitizen` | int (0/1) | 0 = No, 1 = Yes | Whether customer is 65+ years old. **Note:** stored as integer but it's conceptually categorical |
| `Partner` | categorical | Yes, No | Whether customer has a spouse/partner |
| `Dependents` | categorical | Yes, No | Whether customer has dependents (children, elderly parents, etc.) living with them |

**Why this matters:** Demographics help segment customers. Family customers (Partner=Yes, Dependents=Yes) often have multi-line plans, are stickier, churn less. Single seniors may have different needs than young singles.

### Account information (3 columns)
| Column | Type | Values | What it really means |
|---|---|---|---|
| `tenure` | int | 0 to 72 | **Months the customer has been with the company.** A `tenure=1` is a brand-new customer; `tenure=72` has been around 6 years |
| `Contract` | categorical | Month-to-month, One year, Two year | Contract length they signed up for |
| `PaperlessBilling` | categorical | Yes, No | Whether they get e-bills vs paper bills mailed |

**Why this matters — pay attention here, this is THE most important section:**

- **Tenure is THE strongest churn signal in this dataset.** New customers churn at very high rates; long-tenured customers almost never leave. This is "survivorship": the customers still around at month 60 are the ones who already proved they're sticky.
- **Contract type is the second strongest signal.** Someone on a 2-year contract literally cannot leave without paying an early termination fee. Month-to-month customers can walk away anytime — and they do.
- **Paperless billing** correlates with younger, more tech-savvy customers, who churn more.

### Phone services (2 columns)
| Column | Type | Values | What it really means |
|---|---|---|---|
| `PhoneService` | categorical | Yes, No | Has a landline/phone line with the company |
| `MultipleLines` | categorical | Yes, No, No phone service | Has more than one phone line (e.g., business + personal) |

**Note the redundancy:** if `PhoneService = No`, then `MultipleLines = "No phone service"` necessarily. This is **logical dependency** — useful to know for feature engineering.

### Internet services (7 columns)
| Column | Type | Values | What it really means |
|---|---|---|---|
| `InternetService` | categorical | DSL, Fiber optic, No | Type of internet service. **DSL = older, slower, cheaper. Fiber = newer, faster, more expensive.** "No" means no internet at all |
| `OnlineSecurity` | categorical | Yes, No, No internet service | Add-on: virus/malware protection |
| `OnlineBackup` | categorical | Yes, No, No internet service | Add-on: cloud backup |
| `DeviceProtection` | categorical | Yes, No, No internet service | Add-on: hardware insurance |
| `TechSupport` | categorical | Yes, No, No internet service | Add-on: premium tech support |
| `StreamingTV` | categorical | Yes, No, No internet service | Add-on: TV streaming bundle |
| `StreamingMovies` | categorical | Yes, No, No internet service | Add-on: movie streaming bundle |

**Why this matters — another critical section:**

- These 6 add-ons (`OnlineSecurity` through `StreamingMovies`) are all "you can only have this if you have internet" services. So they all share the third category `"No internet service"`.
- **Fiber optic customers churn at surprisingly high rates** — counterintuitive, but real. The hypothesis is that fiber is expensive, customers expect premium service, and when service isn't perfect they leave faster than DSL customers who knew they were getting budget service.
- **Customers with more add-ons (especially TechSupport and OnlineSecurity) churn less.** Why? Because (a) they're more invested in the service, (b) switching providers means re-setting up all those add-ons elsewhere = friction.
- **Streaming add-ons (TV + Movies) are weaker churn signals** than security/support, possibly because streaming services from Netflix/Disney/etc. are easily replaceable.

### Billing (3 columns)
| Column | Type | Values | What it really means |
|---|---|---|---|
| `PaymentMethod` | categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | How they pay |
| `MonthlyCharges` | float | ~$18 to ~$120 | What they're billed each month — sum of all services |
| `TotalCharges` | float | ~$18 to ~$8,700 | Total ever paid (≈ tenure × MonthlyCharges) |

**Why this matters:**

- **`PaymentMethod` is one of the strongest churn signals in this dataset**, and it's not obvious why at first. **Electronic check customers churn dramatically more** than auto-pay (bank transfer / credit card) customers. The hypothesis: e-check is the "manual, distrustful, easy-to-cancel" option. Auto-pay customers have set-and-forget billing and just keep paying. Behavioral finance > demographics, often.
- **`MonthlyCharges` higher = higher churn risk**, particularly for customers without long contracts. Pricey month-to-month is the danger zone.
- **`TotalCharges` is essentially redundant** with tenure × MonthlyCharges. It exists in the dataset but you should be careful about including all three in a linear model.

### Target
| Column | Type | Values | What it really means |
|---|---|---|---|
| `Churn` | categorical | Yes, No | **Did this customer cancel in the last month?** This is what you're predicting |

The target is defined as: **churn within the most recent month of observation**, not over the whole tenure. So `Churn=Yes` for a customer with `tenure=72` means they were with the company for 6 years and then left in the last month.

---

## The known data quality issue

**`TotalCharges` is stored as a string in the CSV.** Why? Because 11 rows contain a literal whitespace character `" "` instead of a number. Pandas reads the whole column as `object` to accommodate them.

You already found this. The question Q3 wants you to answer is **why are those 11 rows blank** — there's a perfectly logical business reason once you look at the rows. That's the detective work.

---

## Mental model for the project

Before you do any more analysis, internalize this hierarchy of expected importance (based on domain knowledge — not yet your data):

**Strong churn signals (you should expect these to dominate):**
- `tenure` (very strong — new customers leave most)
- `Contract` (very strong — month-to-month customers can leave)
- `PaymentMethod` (strong — electronic check is high-risk)
- `InternetService` (strong — fiber customers churn more)

**Moderate signals:**
- `OnlineSecurity`, `TechSupport` (stickiness through add-ons)
- `MonthlyCharges` (higher bill = higher cancellation pressure)
- `PaperlessBilling`

**Weak / probably noise:**
- `gender` (should have ≈ zero predictive value — if your model thinks gender is important, something is wrong)
- `PhoneService` (most customers have it; not very discriminating)
- `StreamingTV`, `StreamingMovies` (mild at best)

**Building this expectation BEFORE looking at the data** is what separates analysts from script-runners. Your job in EDA is to **confirm or refute** these hypotheses with evidence. If gender turns out to be predictive in your model, that's a red flag worth investigating — not a finding.

---
