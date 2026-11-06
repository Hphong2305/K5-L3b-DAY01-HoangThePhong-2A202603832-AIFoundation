import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "src" / "day01_spam_ml_dl_embedding_llm.ipynb"

STUDENT_TODOS = [
    "vectorize",
    "sigmoid",
    "predict_proba",
    "train_logreg",
    "metrics",
    "build_mlp",
    "train_mlp",
    "cosine_similarity",
    "top_k_similar",
    "knn_predict",
    "call_openai",
    "build_classification_prompt",
    "parse_label",
    "llm_classify_batch",
]

PREDICTION_KEYS = [
    "ml_f1", "ml_lr", "dl_curve", "dl_relu",
    "emb_search", "emb_tricky", "llm_models", "llm_fewshot", "tradeoff", "hard_eval",
]


class RepositoryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [c for c in cls.cells if c.get("cell_type") == "code"]
        cls.code_source = "\n".join("".join(c.get("source", [])) for c in cls.code_cells)
        cls.all_source = "\n".join("".join(c.get("source", [])) for c in cls.cells)

    def test_notebook_is_valid_and_has_no_stored_runtime_output(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        ids = [c.get("id") for c in self.cells]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(self.code_cells)
        self.assertTrue(all(c.get("execution_count") is None for c in self.code_cells))
        self.assertTrue(all(c.get("outputs") == [] for c in self.code_cells))

    def test_python_cells_parse_after_removing_colab_magic(self):
        for i, cell in enumerate(self.code_cells):
            src = "\n".join(
                line for line in "".join(cell["source"]).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            with self.subTest(cell=i):
                ast.parse(src)

    def test_student_todos_are_present_and_unsolved(self):
        for name in STUDENT_TODOS:
            with self.subTest(function=name):
                self.assertRegex(self.code_source, rf"def {name}\(")
        self.assertGreaterEqual(self.code_source.count("raise NotImplementedError"), len(STUDENT_TODOS) - 2)
        self.assertIn('STUDENT_ID = ""', self.code_source)
        self.assertIn('"baseline_accuracy": None', self.code_source)

    def test_no_solution_helpers_or_secrets_leak(self):
        self.assertNotIn("_FakeOpenAI", self.code_source)
        self.assertNotIn("sk-fake", self.code_source)
        self.assertIsNone(re.search(r"sk-[A-Za-z0-9_-]{20,}", self.all_source))

    def test_anti_vibe_mechanics_are_wired(self):
        for key in PREDICTION_KEYS:
            with self.subTest(prediction=key):
                self.assertIn(f'require_prediction("{key}")', self.code_source)
                self.assertIn(f'predict("{key}"', self.code_source)
        self.assertIn("_no_sklearn", self.code_source)
        self.assertIn("CHALLENGES", self.code_source)
        self.assertEqual(len(re.findall(r'"[0-9a-f]{12}"', self.code_source.split("_H = ")[1].splitlines()[0])), 5)

    def test_dataset_source_and_report_filename(self):
        self.assertIn("uciml/sms-spam-collection-dataset", self.code_source)
        self.assertIn("archive.ics.uci.edu", self.code_source)
        self.assertIn("lab1_report.json", self.all_source)
        self.assertNotIn('"student_id"', self.code_source)

    def test_hard_eval_dataset_is_complete_and_embedded(self):
        import csv
        from collections import Counter
        with open(ROOT / "src" / "data" / "hard_eval_50.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 50)
        self.assertEqual([int(r["id"]) for r in rows], list(range(1, 51)))
        self.assertTrue(all(r["label"] in {"spam", "ham"} and r["text"].strip() and r["why_tricky"].strip() for r in rows))
        self.assertEqual(Counter(r["category"] for r in rows), Counter({
            "ham_looks_spam": 10, "natural_spam": 10, "manipulation": 8, "obfuscation": 6,
            "prompt_injection": 6, "transactional_ham": 6, "talks_about_spam": 4}))
        self.assertEqual(len({r["text"] for r in rows}), 50)
        self.assertIn("HARD_EVAL = pd.DataFrame", self.code_source)
        for r in rows[:3] + rows[-3:]:
            self.assertIn(json.dumps(r["text"], ensure_ascii=False)[1:-1], self.code_source)

    def test_llm_providers_and_claude_path(self):
        for provider in ["openai", "anthropic", "gemini", "groq", "openrouter", "ollama"]:
            with self.subTest(provider=provider):
                self.assertIn(f'"{provider}":', self.code_source)
        self.assertIn('LLM_PROVIDER = "openai"', self.code_source)
        self.assertIn("def call_claude(", self.code_source)
        self.assertIn("def call_llm(", self.code_source)
        self.assertIn("def build_defended_prompt(", self.code_source)
        claude_src = self.code_source.split("def call_claude(")[1].split("def call_llm(")[0]
        self.assertIn("import anthropic", claude_src)
        self.assertNotIn("temperature=", claude_src)
        self.assertIn('fallbacks="default"', claude_src)
        self.assertIn("anthropic", self.code_source.splitlines()[0] + (ROOT / "src" / "requirements.txt").read_text())


if __name__ == "__main__":
    unittest.main()
