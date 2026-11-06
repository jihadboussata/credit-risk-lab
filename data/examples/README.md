# Synthetic examples

`synthetic_accounts.csv` contains three invented account profiles, not source
customers. Each repeats a chosen bill, payment and repayment status over six
months. These simplified combinations are for software demonstrations, not an
economic simulation or a representation of real customers.

Amounts use the source dataset's NTD units. Audit fields sex and age are included
to match the raw schema but are excluded by the model's explicit feature allowlist.
See `docs/data_dictionary.md` for field definitions and source-code ambiguities.
