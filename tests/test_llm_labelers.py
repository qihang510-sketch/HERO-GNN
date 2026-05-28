import json

import pytest

from scripts.build_llm_labels import main as build_labels_main
from scripts.run_labeler_comparison import build_comparison_rows
from src.data.synthetic_builder import generate_synthetic_dataset
from src.llm.base_labeler import (
    OPTIONAL_REAL_LLM_MESSAGE,
    OptionalLabelerUnavailable,
    build_risk_card_prompt,
    normalize_label,
)
from src.llm.local_qwen_labeler import LocalQwenRiskLabeler
from src.llm.openai_labeler import OpenAIRiskLabeler
from src.training.trainer import train_single_experiment


def test_prompt_and_label_normalization():
    card = {
        "target_id": "r0",
        "neighbor_id": "r1",
        "metapath": "review-user-review",
        "semantic_similarity": 0.2,
        "structural_score": 0.9,
    }
    messages = build_risk_card_prompt(card)
    assert "Return JSON only" in messages[0]["content"]
    assert "Risk card" in messages[1]["content"]

    label = normalize_label(
        {
            "mechanism": "not_valid",
            "risk_relevance": 2,
            "confidence": 3.0,
            "rationale": "x",
        },
        risk_card=card,
    )
    assert label == {
        "target_id": "r0",
        "neighbor_id": "r1",
        "metapath": "review-user-review",
        "mechanism": "irrelevant_heterophily",
        "risk_relevance": 0,
        "confidence": 1.0,
        "rationale": "x",
    }


def test_real_labelers_fail_with_friendly_optional_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_QWEN_MODEL_PATH", raising=False)
    with pytest.raises(OptionalLabelerUnavailable, match=OPTIONAL_REAL_LLM_MESSAGE):
        OpenAIRiskLabeler()
    with pytest.raises(OptionalLabelerUnavailable, match=OPTIONAL_REAL_LLM_MESSAGE):
        LocalQwenRiskLabeler()


def test_build_mock_labels_and_compare(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=tmp_path / "synthetic",
        processed_dir=processed_dir,
        num_reviews=120,
        num_users=25,
        num_items=12,
        num_devices=18,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=8,
        text_dim=12,
    )
    mock_file = tmp_path / "llm_labels_mock.jsonl"
    openai_file = tmp_path / "llm_labels_openai.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_llm_labels.py",
            "--dataset",
            "synthetic",
            "--data_dir",
            str(processed_dir),
            "--labeler",
            "mock",
            "--max_cards",
            "8",
            "--out_file",
            str(mock_file),
            "--seed",
            "8",
        ],
    )
    build_labels_main()

    lines = mock_file.read_text(encoding="utf-8").strip().splitlines()
    assert 0 < len(lines) <= 8
    first = json.loads(lines[0])
    assert set(first) == {"target_id", "neighbor_id", "metapath", "mechanism", "risk_relevance", "confidence", "rationale"}

    openai_file.write_text(mock_file.read_text(encoding="utf-8"), encoding="utf-8")
    rows = build_comparison_rows("synthetic", [mock_file, openai_file])
    assert {row["labeler"] for row in rows} == {"mock", "openai"}
    assert all(row["agreement_with_mock"] == 1.0 for row in rows)


def test_hero_can_use_prebuilt_llm_label_file(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed" / "synthetic"
    generate_synthetic_dataset(
        out_dir=tmp_path / "synthetic",
        processed_dir=processed_dir,
        num_reviews=120,
        num_users=25,
        num_items=12,
        num_devices=18,
        fraud_ratio=0.2,
        hetero_fraud_ratio=0.5,
        seed=9,
        text_dim=12,
    )
    label_file = tmp_path / "llm_labels_mock.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_llm_labels.py",
            "--dataset",
            "synthetic",
            "--data_dir",
            str(processed_dir),
            "--labeler",
            "mock",
            "--max_cards",
            "12",
            "--out_file",
            str(label_file),
            "--seed",
            "9",
        ],
    )
    build_labels_main()

    metrics = train_single_experiment(
        dataset="synthetic",
        model_name="hero_gnn",
        seed=9,
        data_dir=processed_dir,
        output_root=tmp_path / "outputs",
        epochs=1,
        hidden_dim=8,
        top_k=3,
        max_candidates_per_node=5,
        homophilic_topk=3,
        heterophilic_topk=3,
        topk_chains=1,
        max_chain_length=1,
        min_chain_quality=0.0,
        llm_label_file=label_file,
    )
    assert metrics["external_llm_label_file"] == str(label_file)
    assert metrics["external_llm_labels_used"] > 0
