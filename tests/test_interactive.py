"""Tests for hybrid interactive mode helpers in main.py."""

from unittest.mock import patch

import pytest

from agent.models import CompanyProfile, GraphExtractionResult


def _make_result(confidence_scores: dict, **profile_overrides) -> GraphExtractionResult:
    defaults = {
        "industry": "Financial Services",
        "company_size": "enterprise",
        "geography": None,
        "tech_stack": ["Oracle DB", "Java EE"],
        "data_maturity": "medium",
        "ai_experience": "none",
        "engineering_team_size": 50,
        "has_ml_engineers": True,
        "budget_tier": "growth",
        "ai_ambition": "improve operational efficiency with AI automation",
        "pain_points": ["Legacy systems slow us down"],
        "compliance_requirements": ["RBI", "PCI-DSS"],
        "timeline_pressure": "experimental",
        "existing_ai_initiatives": None,
        "competitors_doing": None,
    }
    defaults.update(profile_overrides)
    profile = CompanyProfile(**defaults)
    return GraphExtractionResult(
        extracted_profile=profile,
        confidence_scores=confidence_scores,
        low_confidence_fields=[f for f, c in confidence_scores.items() if c < 0.6],
        extracted_entities=[],
        extraction_warnings=[],
    )


class TestAskOrPrefillText:
    def test_offers_prefill_as_default_when_above_threshold(self):
        from main import _ask_or_prefill_text

        result = _make_result({"industry": 0.85})
        with patch("main.Prompt.ask", return_value="Financial Services") as mock_ask:
            _ask_or_prefill_text("industry", result)
            _, kwargs = mock_ask.call_args
            assert kwargs.get("default") == "Financial Services"

    def test_user_override_replaces_prefill(self):
        from main import _ask_or_prefill_text

        result = _make_result({"industry": 0.85})
        with patch("main.Prompt.ask", return_value="Retail"):
            val = _ask_or_prefill_text("industry", result)
        assert val == "Retail"

    def test_no_prefill_when_confidence_below_threshold(self):
        from main import _ask_or_prefill_text

        result = _make_result({"industry": 0.55})
        with patch("main.Prompt.ask", return_value="Retail") as mock_ask:
            _ask_or_prefill_text("industry", result)
            _, kwargs = mock_ask.call_args
            assert "default" not in kwargs

    def test_no_prefill_when_result_is_none(self):
        from main import _ask_or_prefill_text

        with patch("main.Prompt.ask", return_value="Healthcare") as mock_ask:
            _ask_or_prefill_text("industry", None)
            _, kwargs = mock_ask.call_args
            assert "default" not in kwargs

    def test_always_ask_field_never_prefilled_even_at_full_confidence(self):
        from main import ALWAYS_ASK, _ask_or_prefill_text

        assert "ai_ambition" in ALWAYS_ASK
        result = _make_result({"ai_ambition": 1.0}, ai_ambition="reduce operational costs using AI")
        with patch("main.Prompt.ask", return_value="my own answer") as mock_ask:
            _ask_or_prefill_text("ai_ambition", result)
            _, kwargs = mock_ask.call_args
            assert "default" not in kwargs


class TestAskOrPrefillBool:
    def test_uses_prefill_as_default_when_above_threshold(self):
        from main import _ask_or_prefill_bool

        result = _make_result({"has_ml_engineers": 0.8}, has_ml_engineers=True)
        with patch("main.Confirm.ask", return_value=True) as mock_confirm:
            _ask_or_prefill_bool("has_ml_engineers", result)
            _, kwargs = mock_confirm.call_args
            assert kwargs.get("default") is True

    def test_false_prefill_propagated_as_default(self):
        from main import _ask_or_prefill_bool

        result = _make_result({"has_ml_engineers": 0.75}, has_ml_engineers=False)
        with patch("main.Confirm.ask", return_value=False) as mock_confirm:
            _ask_or_prefill_bool("has_ml_engineers", result)
            _, kwargs = mock_confirm.call_args
            assert kwargs.get("default") is False

    def test_no_default_when_confidence_below_threshold(self):
        from main import _ask_or_prefill_bool

        result = _make_result({"has_ml_engineers": 0.4}, has_ml_engineers=True)
        with patch("main.Confirm.ask", return_value=False) as mock_confirm:
            _ask_or_prefill_bool("has_ml_engineers", result)
            _, kwargs = mock_confirm.call_args
            assert "default" not in kwargs

    def test_no_prefill_when_result_is_none(self):
        from main import _ask_or_prefill_bool

        with patch("main.Confirm.ask", return_value=False) as mock_confirm:
            _ask_or_prefill_bool("has_ml_engineers", None)
            _, kwargs = mock_confirm.call_args
            assert "default" not in kwargs


class TestAskOrPrefillList:
    def test_returns_prefill_list_when_accepted(self):
        from main import _ask_or_prefill_list

        result = _make_result({"compliance_requirements": 0.9})
        with patch("main.Confirm.ask", return_value=True):
            val = _ask_or_prefill_list("compliance_requirements", result)
        assert val == ["RBI", "PCI-DSS"]

    def test_falls_through_to_manual_when_rejected(self):
        from main import _ask_or_prefill_list

        result = _make_result({"compliance_requirements": 0.9})
        with patch("main.Confirm.ask", return_value=False):
            with patch("main.Prompt.ask", side_effect=["GDPR", ""]):
                val = _ask_or_prefill_list("compliance_requirements", result)
        assert val == ["GDPR"]

    def test_always_ask_field_goes_straight_to_manual(self):
        from main import ALWAYS_ASK, _ask_or_prefill_list

        assert "pain_points" in ALWAYS_ASK
        result = _make_result({"pain_points": 1.0})
        with patch("main.Prompt.ask", side_effect=["Slow reporting", ""]):
            val = _ask_or_prefill_list("pain_points", result)
        assert val == ["Slow reporting"]

    def test_no_prefill_when_confidence_below_threshold(self):
        from main import _ask_or_prefill_list

        result = _make_result({"compliance_requirements": 0.5})
        with patch("main.Prompt.ask", side_effect=["SOC2", ""]):
            val = _ask_or_prefill_list("compliance_requirements", result)
        assert val == ["SOC2"]


class TestAskOrPrefillChoice:
    def test_uses_prefill_key_as_default(self):
        from main import _ask_or_prefill_choice

        result = _make_result({"data_maturity": 0.75}, data_maturity="high")
        dm_map = {"1": "low", "2": "medium", "3": "high"}
        with patch("main.Prompt.ask", return_value="3") as mock_ask:
            val = _ask_or_prefill_choice("data_maturity", result, ["1", "2", "3"], dm_map)
        assert val == "high"
        _, kwargs = mock_ask.call_args
        assert kwargs.get("default") == "3"

    def test_user_override_on_choice_field(self):
        from main import _ask_or_prefill_choice

        result = _make_result({"data_maturity": 0.75}, data_maturity="high")
        dm_map = {"1": "low", "2": "medium", "3": "high"}
        with patch("main.Prompt.ask", return_value="1"):
            val = _ask_or_prefill_choice("data_maturity", result, ["1", "2", "3"], dm_map)
        assert val == "low"

    def test_no_default_when_confidence_below_threshold(self):
        from main import _ask_or_prefill_choice

        result = _make_result({"data_maturity": 0.5}, data_maturity="high")
        dm_map = {"1": "low", "2": "medium", "3": "high"}
        with patch("main.Prompt.ask", return_value="2") as mock_ask:
            _ask_or_prefill_choice("data_maturity", result, ["1", "2", "3"], dm_map)
            _, kwargs = mock_ask.call_args
            assert "default" not in kwargs

    def test_no_default_when_result_is_none(self):
        from main import _ask_or_prefill_choice

        dm_map = {"1": "low", "2": "medium", "3": "high"}
        with patch("main.Prompt.ask", return_value="2") as mock_ask:
            _ask_or_prefill_choice("data_maturity", None, ["1", "2", "3"], dm_map)
            _, kwargs = mock_ask.call_args
            assert "default" not in kwargs


class TestRunDocIngestion:
    def test_returns_none_when_no_paths_entered(self, tmp_path):
        from main import _run_doc_ingestion

        with patch("main.Prompt.ask", return_value=""):
            result = _run_doc_ingestion()
        assert result is None

    def test_skips_nonexistent_file_gracefully(self, tmp_path):
        from main import _run_doc_ingestion

        with patch("main.Prompt.ask", side_effect=["/nonexistent/path.pdf", ""]):
            result = _run_doc_ingestion()
        assert result is None

    def test_loads_valid_text_file(self, tmp_path):
        from main import _run_doc_ingestion

        doc = tmp_path / "audit.txt"
        doc.write_text("Oracle DB used for core banking. Java EE middleware. Team of 40 engineers.")

        fake_result = _make_result({"tech_stack": 0.9})
        with patch("main.Prompt.ask", side_effect=[str(doc), ""]):
            with patch("ingestion.graph_builder.build_graph") as mock_graph:
                with patch("ingestion.extractor.extract_profile", return_value=fake_result):
                    result = _run_doc_ingestion()
        assert result is fake_result
