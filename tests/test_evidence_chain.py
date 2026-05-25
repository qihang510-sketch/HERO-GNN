from src.graph.evidence_chain import build_evidence_chain


def test_evidence_chain_ranks_scores():
    chain = build_evidence_chain(0, {1: 0.2, 2: 0.9}, max_length=1)
    assert chain == [{"source": 0, "target": 2, "score": 0.9}]

