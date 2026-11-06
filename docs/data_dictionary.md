# Data dictionary and handling decisions

The canonical input has 25 columns: one identifier, 23 source predictors, and one target. Names are lowercased for consistent Python/SQL use. The original dataset is credited in [references.md](references.md). Amounts are in New Taiwan dollars (NT$), not US dollars.

| Canonical column(s) | Meaning | Modeling treatment |
|---|---|---|
| `id` | Source account identifier | Audit/split key; excluded from predictors |
| `limit_bal` | Credit limit | Numeric |
| `sex` | Source codes: 1 male, 2 female | Audit only; not a predictor |
| `education` | 1 graduate school, 2 university, 3 high school, 4 other | Categorical; other observed codes consolidated to unspecified (0) |
| `marriage` | 1 married, 2 single, 3 other | Categorical; additional observed codes retained distinctly |
| `age` | Age in years | Audit only; not a predictor |
| `pay_0`, `pay_2` … `pay_6` | Repayment status, September to April 2005 | Categorical; retain original values |
| `bill_amt1` … `bill_amt6` | Monthly bill, September to April | Numeric; preserve negative balances |
| `pay_amt1` … `pay_amt6` | Monthly payment, September to April | Numeric |
| `target` | Next-month default: 1 yes, 0 no | Outcome; never a predictor |

## Read the six months in the correct order

| Month (2005) | Repayment status | Bill | Payment |
|---|---|---|---|
| September | `pay_0` | `bill_amt1` | `pay_amt1` |
| August | `pay_2` | `bill_amt2` | `pay_amt2` |
| July | `pay_3` | `bill_amt3` | `pay_amt3` |
| June | `pay_4` | `bill_amt4` | `pay_amt4` |
| May | `pay_5` | `bill_amt5` | `pay_amt5` |
| April | `pay_6` | `bill_amt6` | `pay_amt6` |

There is deliberately no canonical `pay_1`: the published September field is `PAY_0`. Renaming it without documenting the change is a common source of off-by-one month errors.

## Codes require care

The published description defines -1 as timely payment and positive values as months of delay, with 9 meaning nine months or more. The actual file also uses -2 and 0. Their meaning is not fully defined in that description. Keep these as distinct status categories; do not silently label them “zero months late,” missing data, or default.

Education codes outside 1–4 are consolidated to an explicit unspecified category (0); marital-status codes remain distinct categories. This is a declared modeling decision, not a recovered source definition. A number stored as an integer is not automatically a meaningful numerical scale. In particular, a model should not assume education code 4 is twice education code 2.

## Missingness and unusual values

The source is advertised without missing cells, but the audit checks this independently. A complete cell is not necessarily a completely specified concept. Ambiguous categories remain a data-quality limitation.

Negative bills are preserved because an account can have a credit balance. Bills can exceed the stated limit. Outlier clipping must have a documented rationale and be learned from training only; it is not applied merely to make a distribution look cleaner. Safe ratios must define their denominators, zero cases, and interpretation. A payment-to-bill ratio is a descriptive feature, not a verified accounting reconciliation across billing cycles.

## Audit and inference boundaries

Keep sex and age for subgroup evaluation; exclude them from primary model inputs. Education, marital status, and payment history may still encode social or demographic differences. This is not a fairness guarantee.

The app uses synthetic profiles for individual scoring. The dataset identifier is a key, not personal identification or a feature. A single historical Taiwanese portfolio cannot support claims about current applicants, other institutions, or future economic conditions without additional data.
