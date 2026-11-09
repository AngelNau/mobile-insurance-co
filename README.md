# Claims-handling agent – mobile device insurance

A LangGraph `StateGraph` agent that handles damage claims for a fictional insurer of phones and laptops.
It is a fixed flow with an explicit state (`ClaimState`). Everything is in [main.py](main.py).

## Graph

![Agent graph](agent_graph.png)

Dotted edges are conditional:
- `intake` → `ask_missing` if data is missing, → `check_policy` if the claim is complete, → `decide` if data is still missing after 3 rounds.
- `check_policy` → `check_coverage` if the policy is valid, → `decide` on a policy error.
- `decide` → `__end__` on APPROVED / REJECTED, → `human_review` on ESCALATION.

The image is generated from the compiled graph with `python main.py mermaid`.

| Node | What it does | LLM? |
|---|---|---|
| `intake` | Extracts policy number, device, date, damage type and amount (Pydantic structured output). Each value is then checked against the claim text so the model can't make up a policy number, date, amount or brand. | yes |
| `ask_missing` | `interrupt()` asks the user for the missing fields (Article 11), then goes back to `intake`. | – |
| `check_policy` | The `check_policy_tool` tool over `policies.json` checks that the policy exists, its status (Art. 12), the coverage period (Art. 2), which device is insured (Art. 1), and the limit, deductible and device age. The device must match the insured model and variant (e.g. a Galaxy A15 or S24 Ultra is not a Galaxy S24), while brand, series word, case and spacing (`S24`, `galaxy s-24`) may differ. | – |
| `check_coverage` | RAG: BM25 retrieval over the coverage articles of [terms.md](terms.md) (Art. 3–6), then the LLM decides and cites an article. If the cited article is not one of the retrieved ones, the model's verdict is discarded and the claim is escalated to an assessor. | yes |
| `decide` | Deterministic decision and payout: `min(claimed, depreciated value, limit) − deductible`, with depreciation of 20 %/year capped at 60 % (Art. 7–9). Escalates if the amount is > €800, the incident happened < 30 days after the policy start (Art. 10), or coverage could not be verified (model error or a citation outside the retrieved context). | – |
| `human_review` | `interrupt()` (checkpointer `MemorySaver`). The assessor approves or rejects in the CLI, then the graph resumes. | – |

Every decision comes with a reason that cites an article of the terms.

## Files

[terms.md](terms.md) and [policies.json](policies.json) are synthetic data made for this task. The insurer, the policyholders and the policies are fictional.

- [terms.md](terms.md) – insurance terms (12 articles)
- [policies.json](policies.json) – 5 policies
- [claims/](claims/) – `C01–C10` provided test claims, `X01–X07` custom cases (X01–X03 adversarial: prompt injection, intentional damage passed off as an accident, theft without a police report; X04 is the exact €800 boundary, X05 a non-existent policy, X06 a Galaxy A15 claim on a Galaxy S24 policy, X07 a claim below the deductible). Each file holds the expected outcome.
- [eval_results.md](eval_results.md) – latest evaluation results
- [tests/](tests/) – pytest tests that run without an LLM

## Running

### With Docker

Needs only Docker with the Compose plugin. Compose starts Ollama 0.40.11, pulls `llama3.2` into a named volume (only on the first run), and runs the agent.

```bash
docker compose run --rm app              # interactive CLI
docker compose run --rm app eval         # evaluation -> eval_results.md
docker compose run --rm app mermaid      # graph diagram -> agent_graph.png
docker compose run --rm app eval --remote
docker compose run --rm --build tests    # pytest, without Ollama (--build picks up changed requirements)
docker compose down                      # stop Ollama (add -v to also delete the downloaded model)
```

The project directory is mounted into the container, so output files land in the repo and `.env` is picked up for `--remote`. To use an NVIDIA GPU, uncomment the `deploy` block in [docker-compose.yml](docker-compose.yml).

### Without Docker

To run the code, you first need to install the dependencies listed in [requirements.txt](requirements.txt).

```bash
ollama pull llama3.2               # local model (default)

python main.py                     # interactive CLI
python main.py eval                # evaluation over claims/ -> eval_results.md
python main.py mermaid             # graph diagram -> agent_graph.png

python main.py eval --remote       # Claude instead of llama3.2
```

For `--remote`, put `ANTHROPIC_API_KEY=...` in a `.env` file.

Dates in claims are accepted as `2026-08-03`, `3 August 2026`, `August 3, 2026` or `3/8/2026` (day first). Amounts as `€250`, `250 €`, `250 EUR` or `1,500 euros`.

## Tests

```bash
python -m pytest
docker compose run --rm --build tests    # the same tests in Docker
```

The tests replace the LLM with `FakeLLM` ([tests/conftest.py](tests/conftest.py)), LangChain's `GenericFakeChatModel` with scripted tool-call answers, so they need neither Ollama nor an API key and are deterministic. They cover:

- **Device check (Art. 1):** a Samsung Galaxy A15 claim on the Galaxy S24 policy is rejected without reaching the coverage step, other models and variants of the same brand are rejected, and reasonable spellings of the insured device are still accepted.
- **Invalid citation:** when the coverage model cites an article that was not in the retrieved context, the claim is escalated to an assessor instead of being decided automatically.

The device spellings are unit tests of `check_policy_tool`; the rest are end-to-end tests through the compiled graph.

## Evaluation

`python main.py eval` runs all 17 cases. Interrupts are answered automatically. The follow-up answers come from the claim file and the assessor's decision comes from `expected.assessor_decision`. For escalated cases, the scored decision is `ESCALATION` (the system's decision before human review). The payout is scored after the assessor approves.

Result with `ollama:llama3.2` (full table in [eval_results.md](eval_results.md)):

| Metric | Result |
|---|---|
| Decision accuracy | 17/17 (100 %) |
| Payout correctness | 17/17 (100 %) |
| Cited article correctness | 17/17 (100 %) |

An earlier version of the project used Slovenian terms and claims. The same pipeline scored 12/15 on all three metrics, and every miss (C04, C06, X02) was in the coverage step. After the terms and claims were translated to English, those cases pass. This is because the 3B model seems to reason about the coverage articles much better in English than in Slovenian.

Only `ollama:llama3.2` has been evaluated. No remote LLM tokens were available, so the `--remote` path is implemented but has not been tested.

### Where the system is weak and why

- **Article 5 depends on the LLM's reasoning.** Theft is covered only with a break-in *and* a police report, so this is the one article where the verdict depends on the model weighing conditions. C04 and C06 failed here with Slovenian input. Extracting these facts as booleans (`break_in`, `police_report`) would let code decide instead.
- **A wrong citation leads to a wrong verdict.** For articles that only grant cover (3, 4) or only exclude it (6), the cited article fixes the verdict, so if the model cites the wrong one, the decision is wrong too. A citation outside the retrieved context is caught and escalated, but a wrong citation *among* the retrieved articles is not.
- **Device matching relies on a hand-written word list.** Model numbers (any token with a digit) and the variant words in `VARIANT_WORDS` (`pro`, `ultra`, `air`, `carbon`, …) must match the policy exactly. A variant word missing from that list is ignored, so e.g. a "Galaxy S24 Neo" would be accepted on a Galaxy S24 policy. A claim that names just the brand ("my Samsung") is rejected under Art. 1 rather than asked about.
- **Retrieval depends on the damage type from intake.** In X02 the claimant threw the phone in anger but asks for it to be reported as a drop. It passes only because intake classifies it as `intentional_damage`. As `physical_damage`, the exclusions article (Art. 6) isn't retrieved and the claim would most likely be approved.
