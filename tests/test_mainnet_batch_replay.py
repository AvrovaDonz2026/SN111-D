from scripts.replay_mainnet_batch import source_map


def test_replay_source_precedence_and_page_aliases_match_validator():
    reference = {
        "spans": [{"span_id": "paper-p001-markdown", "text": "Reference text."}]
    }
    miner = {
        "spans": [{"span_id": "paper-p001-markdown", "text": "Validator miner text."}]
    }
    eligibility = source_map([reference])
    merged = source_map([reference, miner])
    assert eligibility["paper-p001-markdown"] == "Reference text."
    assert merged["paper-p001-markdown"] == "Validator miner text."
    assert merged["paper-p001-001"] == "Reference text."


def test_replay_supports_source_id_and_quote_fallbacks():
    assert source_map(
        [{"spans": [{"id": "s1", "quote": " Evidence "}, {"text": ""}]}]
    ) == {"s1": "Evidence"}
