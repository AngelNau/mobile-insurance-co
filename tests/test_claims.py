"""Tests that run without a real LLM: `main.llm` is replaced by the scripted FakeLLM from conftest.py."""
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
