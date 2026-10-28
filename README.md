# 19 · survey

Weighted survey analysis: frequencies, crosstabs, margins of error.

## Run

```bash
from survey.analysis import Response, weighted_frequencies, crosstab
```

## Test

```bash
pytest projects/19-survey-analyzer -q
```

## What it demonstrates

- Weights are first-class: every calculation accepts them
- Kish effective sample size shows how much weighting costs you
- Chi-square test of independence on weighted counts
- Margins of error with an honest worst-case option
