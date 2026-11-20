# Robustness report

## Benign rewrites and prompt-injection probes (test split)

`flip_rate`: decisions that changed. `harmful_flip_rate`: strong->cheap flips on queries where the cheap model is worse than strong. For injections this should be ~0; a large value means the router can be talked into the cheap model.

| perturbation | kind | mean_abs_dp | mean_dp | flip_rate | to_cheap_rate | to_strong_rate | harmful_flip_rate | retention_after | savings_after |
|---|---|---|---|---|---|---|---|---|---|
| lowercase | benign | 0.002 | -0.001 | 0.011 | 0.005 | 0.005 | 0.005 | 0.969 | 0.094 |
| uppercase | benign | 0.003 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.975 | 0.092 |
| typos_3pct | benign | 0.034 | 0.003 | 0.059 | 0.027 | 0.032 | 0.005 | 0.969 | 0.116 |
| whitespace_pad | benign | 0.000 | -0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.975 | 0.092 |
| polite_prefix | benign | 0.029 | -0.003 | 0.080 | 0.037 | 0.043 | 0.011 | 0.975 | 0.087 |
| polite_suffix | benign | 0.022 | -0.000 | 0.037 | 0.016 | 0.021 | 0.005 | 0.975 | 0.083 |
| irrelevant_filler | benign | 0.037 | -0.005 | 0.091 | 0.032 | 0.059 | 0.011 | 0.982 | 0.129 |
| inject_1 | injection | 0.028 | -0.005 | 0.053 | 0.021 | 0.032 | 0.011 | 0.963 | 0.086 |
| inject_2 | injection | 0.032 | 0.003 | 0.086 | 0.027 | 0.059 | 0.011 | 0.963 | 0.049 |
| inject_3 | injection | 0.031 | 0.001 | 0.053 | 0.016 | 0.037 | 0.000 | 0.982 | 0.069 |
| inject_4 | injection | 0.025 | 0.006 | 0.102 | 0.064 | 0.037 | 0.016 | 0.975 | 0.162 |

**Worst injection harmful-flip rate: 1.60% of queries.** Worst benign flip rate: 9.1%.

## Multilingual probe (4 hand-written parallel questions x 4 languages; a stability check, not a benchmark)

| lang | mean_abs_dp | same_decision |
|---|---|---|
| ar | 0.037 | 0.750 |
| es | 0.027 | 1.000 |
| fr | 0.037 | 0.750 |
| zh | 0.033 | 1.000 |

all-MiniLM-L6-v2 is an English-centric embedder; large shifts here mean you should add multilingual data or route non-English prompts to the strong model by rule.

## Leave-one-domain-out (out-of-distribution)

Trained WITHOUT the held-out domain, tau tuned on the other domains, scored on the whole held-out domain. `ood_meets_target` shows whether quality retention stayed >= 95%.

| held_out_domain | n_ood | ood_savings | ood_retention | ood_pct_cheap | ood_meets_target | indomain_savings | indomain_retention | indomain_pct_cheap |
|---|---|---|---|---|---|---|---|---|
| chat | 136 | 0.050 | 0.984 | 0.066 | True | 0.102 | 1.000 | 0.154 |
| code | 135 | 0.033 | 0.973 | 0.052 | True | 0.000 | 1.000 | 0.000 |
| docqa | 110 | 0.017 | 1.000 | 0.036 | True | 0.082 | 0.895 | 0.136 |
| knowledge | 176 | 0.119 | 0.940 | 0.142 | False | 0.077 | 0.967 | 0.139 |
| math | 259 | 0.000 | 1.000 | 0.000 | True | 0.000 | 1.000 | 0.000 |
| science_qa | 132 | 0.054 | 0.992 | 0.061 | True | 0.269 | 0.920 | 0.308 |

**1 of 6 held-out domains missed the quality target.** Missing the target on an unseen domain is the expected failure mode of a learned router; the mitigation is the uncertain band (`ROUTER_UNCERTAIN_BAND`), hard rules, shadow mode, and retraining when new traffic types appear.
