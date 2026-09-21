
this is simple approach where LLM evaluate the pipeline and trying to fix the error column

# Self-Healing Pipeline

A CSV whose column names differ from what you expect usually kills the pipeline.
This script figures out the right column names first, then carries on with the data.

Say an incoming CSV has `email_address`, but the join needs `customer_email`.
Instead of failing, the column gets renamed first.

## Running it

```
pip install -r requirements.txt
python pipeline.py
```

## How it works

Every CSV in `data/raw` is handled one at a time. Its column names are matched
against the columns of the `sales` table in SQLite, the wrong ones get renamed,
and the result is written to `data/healed`. After that every row is checked: is
the email valid, is the amount actually a number, does the date parse. Broken
rows are moved to `data/quarantine` along with the reason. The rest are joined
to `data/ref/customers.csv` to pick up the customer name and city, written to
`data/output`, and loaded into the `sales` table.

The healed, output and quarantine folders are cleared on every run, so you can
run it as often as you like. If a CSV has columns that can't be figured out,
that file is skipped and the others still go through.

## What's in data/raw

Five samples, each in a different state:

- `transactions.csv` - one column misnamed, and it happens to be the join key
- `sales_ok.csv` - everything already correct
- `sales_drifted.csv` - all four columns misnamed
- `sales_messy.csv` - misnamed columns plus some broken values
- `sales_alien.csv` - columns in Indonesian

## Guessing the column names

the script asks an LLM (Ollama Cloud). That's what
handles Indonesian columns like `nilai` turning into `purchase_amount`.
