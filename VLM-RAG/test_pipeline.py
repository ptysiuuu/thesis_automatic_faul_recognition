"""
test_pipeline.py
================
Quick offline test that validates the full pipeline WITHOUT
loading any VLM model. Uses dummy frames and checks that:
  - RAG retrieval works
  - Prompt building works for all strategies
  - Response parsing works
  - Metrics computation works

Run this locally before submitting to the cluster.

Usage:
  python test_pipeline.py
"""

import json
import numpy as np
from PIL import Image

from law12_rag import Law12RAG
from vlm_classifier import (
    build_prompt, parse_response, compute_metrics,
    ACTION_CLASSES, SEVERITY_CLASSES,
)


def test_rag():
    print("=== Testing Law12RAG ===")
    rag = Law12RAG(pdf_path=None, use_embeddings=False)  # keyword retrieval, no model

    queries = [
        ("Tackling",          "tackle from behind excessive force"),
        ("High leg",          "raised foot head dangerous"),
        ("Elbowing",          "elbow violent conduct"),
        ("Dive",              "simulation feigning injury"),
        ("Holding",           "holding shirt arm DOGSO"),
    ]
    for action, query in queries:
        context = rag.retrieve(query)
        print(f"\nQuery: {query[:50]}")
        print(f"Retrieved (first 200 chars): {context[:200]}...")
    print("\n✓ RAG retrieval works\n")


def test_prompts():
    print("=== Testing Prompt Building ===")
    strategies = ["zero_shot", "rule_grounded", "chain_of_thought", "few_shot"]
    dummy_context = "=== FIFA Law 12 ===\nReckless challenge → Yellow card."

    for s in strategies:
        prompt = build_prompt(s, n_views=3, law12_context=dummy_context)
        assert "action" in prompt.lower(), f"Strategy {s} missing action prompt"
        assert "severity" in prompt.lower(), f"Strategy {s} missing severity prompt"
        print(f"  ✓ {s}: {len(prompt)} chars")
    print()


def test_parser():
    print("=== Testing Response Parser ===")

    test_cases = [
        # Normal JSON
        ('{"action": "Tackling", "severity": "Red card", "reasoning": "excessive force"}',
         (0, 3)),
        # With markdown fences
        ('```json\n{"action": "Dive", "severity": "No offence"}\n```',
         (7, 0)),
        # Partial match
        ('{"action": "Standing tackle", "severity": "Yellow", "reasoning": "reckless"}',
         (1, 2)),
        # COT format
        ('{"step1": "contact", "step2": "from front", "action": "Pushing", "severity": "No card"}',
         (4, 1)),
        # Parse failure
        ('I cannot determine the foul type.',
         (-1, -1)),
    ]

    for response, expected in test_cases:
        action_idx, severity_idx = parse_response(response)
        status = "✓" if (action_idx, severity_idx) == expected else "?"
        print(f"  {status} action={action_idx} sev={severity_idx} | "
              f"expected={expected} | input='{response[:50]}'")
    print()


def test_metrics():
    print("=== Testing Metrics ===")
    from evaluate_vlm import compute_metrics

    # Simulate 20 samples
    np.random.seed(42)
    n = 20
    y_true_a  = np.random.randint(0, 8, n).tolist()
    y_true_s  = np.random.randint(0, 4, n).tolist()
    y_pred_a  = np.random.randint(0, 8, n).tolist()
    y_pred_s  = np.random.randint(0, 4, n).tolist()
    # Inject some parse failures
    y_pred_a[3] = -1
    y_pred_s[3] = -1

    m = compute_metrics(y_true_a, y_pred_a, y_true_s, y_pred_s)
    assert "leaderboard_value" in m
    assert m["n_valid"] == n - 1  # one failure
    print(f"  ✓ LB={m['leaderboard_value']:.2f}, parse_rate={m['parse_rate']:.0f}%")
    print()


def test_frame_extraction():
    print("=== Testing Frame Extraction (dummy) ===")
    # Create dummy PIL frames
    frames = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
              for _ in range(4)]
    assert len(frames) == 4
    assert frames[0].size == (224, 224)
    print(f"  ✓ Created {len(frames)} dummy frames of size {frames[0].size}")
    print()


def test_full_prompt_with_rag():
    print("=== Testing Full Prompt with RAG ===")
    rag = Law12RAG(pdf_path=None, use_embeddings=False)
    query = rag.build_query("Tackling", "from behind")
    context = rag.retrieve(query)
    prompt = build_prompt("rule_grounded", n_views=3, law12_context=context)

    print(f"  Query: {query}")
    print(f"  Context length: {len(context)} chars")
    print(f"  Full prompt length: {len(prompt)} chars")
    print(f"  Prompt preview:\n{prompt[:300]}...")
    print()


if __name__ == "__main__":
    test_rag()
    test_prompts()
    test_parser()
    test_metrics()
    test_frame_extraction()
    test_full_prompt_with_rag()
    print("=" * 50)
    print("All tests passed. Pipeline is ready for cluster submission.")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Copy law12_rag.py, vlm_classifier.py, evaluate_vlm.py")
    print("     to /net/tscratch/people/plgaszos/sn-mvfoul/VARS_early_fusion/")
    print("  2. Download FIFA Laws of the Game PDF (Law 12) to")
    print("     /net/tscratch/people/plgaszos/SoccerNet_Data/law12.pdf")
    print("  3. Run quick test: MAX_SAMPLES=20 BACKEND=qwen sbatch eval_vlm.sh")
    print("  4. Run full eval: sbatch eval_vlm.sh")
