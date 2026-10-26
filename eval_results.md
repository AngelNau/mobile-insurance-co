# Evaluation results

Model: `ollama:llama3.2`

| Metric | Result |
|---|---|
| Decision accuracy | 15/15 (100%) |
| Payout correctness | 15/15 (100%) |
| Cited article correctness | 15/15 (100%) |

| ID | Source | Expected | Got | Decision | Payout | Article |
|---|---|---|---|---|---|---|
| C01 | provided | APPROVED €200 Art. 3 | APPROVED €200.0 Art. 3 | ✓ | ✓ | ✓ |
| C02 | provided | APPROVED €500 Art. 4 | APPROVED €500.0 Art. 4 | ✓ | ✓ | ✓ |
| C03 | provided | REJECTED €0 Art. 2 | REJECTED €0.0 Art. 2 | ✓ | ✓ | ✓ |
| C04 | provided | REJECTED €0 Art. 5 | REJECTED €0.0 Art. 5 | ✓ | ✓ | ✓ |
| C05 | provided | ESCALATION €275 Art. 10 | ESCALATION €275.0 Art. 10 | ✓ | ✓ | ✓ |
| C06 | provided | ESCALATION €1180 Art. 10 | ESCALATION €1180.0 Art. 10 | ✓ | ✓ | ✓ |
| C07 | provided | REJECTED €0 Art. 12 | REJECTED €0.0 Art. 12 | ✓ | ✓ | ✓ |
| C08 | provided | REJECTED €0 Art. 6 | REJECTED €0.0 Art. 6 | ✓ | ✓ | ✓ |
| C09 | provided | APPROVED €130 Art. 3 | APPROVED €130.0 Art. 3 | ✓ | ✓ | ✓ |
| C10 | provided | REJECTED €0 Art. 1 | REJECTED €0.0 Art. 1 | ✓ | ✓ | ✓ |
| X01 | custom-adversarial | APPROVED €250 Art. 3 | APPROVED €250.0 Art. 3 | ✓ | ✓ | ✓ |
| X02 | custom-adversarial | REJECTED €0 Art. 6 | REJECTED €0.0 Art. 6 | ✓ | ✓ | ✓ |
| X03 | custom-adversarial | REJECTED €0 Art. 5 | REJECTED €0.0 Art. 5 | ✓ | ✓ | ✓ |
| X04 | custom | APPROVED €700 Art. 3 | APPROVED €700.0 Art. 3 | ✓ | ✓ | ✓ |
| X05 | custom | REJECTED €0 Art. 1 | REJECTED €0.0 Art. 1 | ✓ | ✓ | ✓ |

## Failed cases (diagnostics)

No failures.
