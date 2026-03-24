"""Tests for facade API changes (deprecations)."""

import warnings

import pytest

from karenina.benchmark import Benchmark


def _create_benchmark():
    return Benchmark.create(name="test_facade_api")


@pytest.mark.unit
class TestFacadeResultDeprecations:
    """Test that facade result methods emit deprecation warnings."""

    def test_store_verification_results_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.store_verification_results({})
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_get_verification_results_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_verification_results()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_get_verification_history_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_verification_history()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_clear_verification_results_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.clear_verification_results()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_export_verification_results_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.export_verification_results()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_export_verification_results_to_file_warns(self, tmp_path):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.export_verification_results_to_file(tmp_path / "out.json")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_load_verification_results_from_file_warns(self, tmp_path):
        b = _create_benchmark()
        # Create a minimal valid results file
        results_file = tmp_path / "results.json"
        results_file.write_text("{}")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.load_verification_results_from_file(results_file)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_get_verification_summary_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_verification_summary()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_get_all_run_names_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_all_run_names()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)

    def test_get_results_statistics_by_run_warns(self):
        b = _create_benchmark()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            b.get_results_statistics_by_run()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "ResultsStore" in str(w[0].message)
