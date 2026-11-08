# Blind human review results

precision@10 = share of a method's top 10 (evaluation episodes 6-50) labelled **Pass** (postable as-is).

| Method | Pass | Uncertain | Fail | precision@10 | lenient (Pass+Uncertain) | lower-ranked sample Pass/n |
|---|---|---|---|---|---|---|
| baseline | 7 | 2 | 1 | 0.70 | 0.90 | 1/4 |
| stage_a | 7 | 3 | 0 | 0.70 | 1.00 | 2/4 |
| stage_a_b | 5 | 5 | 0 | 0.50 | 1.00 | 2/4 |

## Top-10 picks a human failed (failure-case candidates)

- baseline #7: E29_0374_0743 (R006) 

The lower-ranked sample is small; it hints at missed promos but is not a recall estimate.
