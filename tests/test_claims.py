"""Tests that run without a real LLM: `main.llm` is replaced by the scripted FakeLLM from conftest.py."""
import json

import pytest

import main
from conftest import claim_data, coverage, unused_answers


def test_baseline_c01_with_fake_llm(fake_llm, run):
    """Sanity check of the harness: C01 (S24, drop, €250) goes through the whole graph."""
    message = ("Hello, policy number POL-1001. On 10 June 2026 my Samsung Galaxy S24 fell on the pavement "
               "and the screen cracked. The repair costs €250.")
    fake = fake_llm(
        claim_data(policy_number="POL-1001", device="Samsung Galaxy S24", device_type="phone",
                   event_date="2026-06-10", damage_type="physical_damage",
                   description="The phone fell and the screen cracked.", amount=250),
        coverage(3, True),
    )
    r = run(message)
    assert (r["system_decision"], r["payout"], r["article"]) == (main.APPROVED, 200, 3)
    assert unused_answers(fake) == []


# ---------------------------------------------------------------- device check (Article 1)
#
# Expected: a claim for a Samsung Galaxy A15 on POL-1001, which insures a Samsung Galaxy S24, is REJECTED
# under Article 1 with €0, and the coverage step (the second LLM call) is never reached.
# Current behaviour: check_policy_tool only asks whether the claimed device and the insured device share
# *any* word. "Samsung Galaxy A15" and "Samsung Galaxy S24" share "samsung" and "galaxy", so the claim passes
# the device check, goes to coverage and is APPROVED under Article 3 (€200).

A15_MESSAGE = ("Policy POL-1001. On 10 June 2026 my Samsung Galaxy A15 fell on the pavement and the screen "
               "cracked. The repair costs €250.")


def test_a15_claim_on_s24_policy_is_rejected(fake_llm, run):
    fake = fake_llm(
        claim_data(policy_number="POL-1001", device="Samsung Galaxy A15", device_type="phone",
                   event_date="2026-06-10", damage_type="physical_damage",
                   description="The phone fell and the screen cracked.", amount=250),
        # What the model would plausibly answer if the claim got this far.
        coverage(3, True),
    )
    r = run(A15_MESSAGE)
    assert (r["system_decision"], r["payout"], r["article"]) == (main.REJECTED, 0, 1)
    assert "Galaxy A15" in r["reasoning"]
    assert unused_answers(fake) == ["CoverageResult"]


def test_hyphenated_model_in_claim_still_matches_end_to_end(fake_llm, run):
    """The claim writes "S-24", the model normalises it to "S24": intake must not drop it as an unseen word."""
    message = ("Policy POL-1001. On 10 June 2026 my Galaxy S-24 fell on the pavement and the screen cracked. "
               "The repair costs €250.")
    fake_llm(
        claim_data(policy_number="POL-1001", device="Samsung Galaxy S24", device_type="phone",
                   event_date="2026-06-10", damage_type="physical_damage",
                   description="The phone fell and the screen cracked.", amount=250),
        coverage(3, True),
    )
    r = run(message)
    assert (r["system_decision"], r["payout"], r["article"]) == (main.APPROVED, 200, 3)


def _device_errors(policy_number: str, device: str) -> list:
    # The incident is dated on the policy's start date, so the only error the check can report is about the device.
    start = next(p["start"] for p in json.loads(main.POLICIES_PATH.read_text()) if p["number"] == policy_number)
    result = main.check_policy_tool.invoke({
        "policy_number": policy_number, "event_date": start, "device": device, "device_type": "",
    })
    return [article for article, _ in result["errors"]]


@pytest.mark.parametrize("policy_number, device", [
    # Reasonable ways of writing the insured device must still be accepted.
    ("POL-1001", "Samsung Galaxy S24"),
    ("POL-1001", "samsung galaxy s24"),
    ("POL-1001", "S24"),
    ("POL-1001", "Galaxy S-24"),
    ("POL-1001", "Samsung Galaxy S24 5G"),
    ("POL-1001", "Samsung Galaxy S24 128GB"),
    ("POL-1003", "iPhone 15 Pro"),
])
def test_device_spelling_variants_are_accepted(policy_number, device):
    assert _device_errors(policy_number, device) == []


@pytest.mark.parametrize("policy_number, device", [
    # Same brand (and series), different model or variant: not the insured device.
    ("POL-1001", "Samsung Galaxy A15"),
    ("POL-1001", "Samsung Galaxy S23"),
    ("POL-1001", "Samsung Galaxy S24 Ultra"),
    ("POL-1002", "MacBook Pro M2"),
    ("POL-1003", "iPhone 15"),
    ("POL-1004", "Lenovo ThinkPad T14"),
])
def test_other_models_of_the_same_brand_are_rejected(policy_number, device):
    assert _device_errors(policy_number, device) == [1]


# ---------------------------------------------------------------- citation outside the retrieved context
#
# The coverage LLM only sees the articles BM25 retrieved, so a citation of any other article is not grounded
# in what the model read. Expected: the model's verdict is discarded (covered=None), the bad citation is
# recorded, and the claim is ESCALATED to an assessor (Article 10) with that reason. A person decides,
# instead of the system approving or rejecting on a hallucination.
# Current behaviour: check_coverage silently replaces the citation with the top retrieved article
# (Article 3 for a drop), whose role then forces covered=True, so the claim is APPROVED automatically.

C01_MESSAGE = ("Hello, policy number POL-1001. On 10 June 2026 my Samsung Galaxy S24 fell on the pavement and "
               "the screen cracked. The repair costs €250.")
C01_CLAIM = dict(policy_number="POL-1001", device="Samsung Galaxy S24", device_type="phone",
                 event_date="2026-06-10", damage_type="physical_damage",
                 description="The phone fell and the screen cracked.", amount=250)


def _retrieved_for_c01() -> list[int]:
    query = f"{main.DAMAGE_QUERY['physical_damage']} {C01_CLAIM['description']} {C01_MESSAGE}"
    return [a["article"] for a in main.retrieve_articles(query)]


def _not_retrieved_coverage_article() -> int:
    return next(a for a in main.COVERAGE_ARTICLES if a not in _retrieved_for_c01())


# not_retrieved: a coverage article (3–6) that exists but was not in the context.
# 7: a real article that is never part of the coverage context. 99: an article that does not exist.
@pytest.mark.parametrize("cited", ["not_retrieved", 7, 99])
def test_citation_outside_context_is_escalated(fake_llm, run, cited):
    cited = _not_retrieved_coverage_article() if cited == "not_retrieved" else cited
    fake_llm(claim_data(**C01_CLAIM),
             coverage(cited, True, f"Covered under Article {cited}."))

    r = run(C01_MESSAGE, assessor="reject")

    assert (r["system_decision"], r["article"]) == (main.ESCALATION, 10)
    assert r["coverage"]["invalid_citation"] == cited
    assert any(f"Article {cited}" in reason and "retrieved" in reason for reason in r["escalation_reasons"])
    # The assessor has the last word: here they reject, so nothing is paid out.
    assert (r["decision"], r["payout"]) == (main.REJECTED, 0)
