"""Base answer class for Karenina.

This module defines the BaseAnswer class, which serves as the foundation for
all answer templates in the benchmark. It provides common functionality and
validation for answer structures.
"""

import inspect
import logging
import re
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Field names that collide with BaseAnswer internal attributes.
# Subclasses that declare any of these as model fields will get a TypeError
# at class definition time, preventing cryptic ValidationErrors at runtime.
_RESERVED_FIELD_NAMES = {"correct"}


class BaseAnswer(BaseModel):
    """Base class for all answer templates in Karenina.

    This class provides common functionality and configuration for answer
    validation and processing.
    """

    model_config = ConfigDict(extra="allow")

    # Question ID will be set programmatically after class instantiation
    id: str | None = None

    # Source code storage (set automatically via __init_subclass__ or manually for exec-created classes)
    # Using ClassVar to prevent Pydantic from treating this as a model field
    _source_code: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically capture source code when Answer classes are defined.

        This hook is called whenever a class inherits from BaseAnswer.
        It attempts to capture the source code using inspect.getsource().
        For exec-created classes, this will fail and _source_code will be None,
        but can be set manually afterwards.
        """
        super().__init_subclass__(**kwargs)
        try:
            cls._source_code = inspect.getsource(cls)
        except (OSError, TypeError):
            # This happens for exec-created classes or when source isn't available
            # The source code can be set manually after class creation
            cls._source_code = None

        # Bridge: ground_truth(self) -> model_post_init(self, __context)
        if "ground_truth" in cls.__dict__ and "model_post_init" not in cls.__dict__:
            original_gt = cls.__dict__["ground_truth"]

            def _bridged_model_post_init(self: Any, __context: Any) -> None:
                original_gt(self)

            cls.model_post_init = _bridged_model_post_init  # type: ignore[method-assign]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-assign verify/verify_granular for VerifiedField templates.

        This hook runs after Pydantic has fully built the model, so
        model_fields is populated and _get_verified_fields() can inspect
        json_schema_extra. Only templates with at least one VerifiedField
        get the auto-generated methods; classic templates are left alone.
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Reject reserved field names that would collide with internal attributes
        reserved_conflicts = _RESERVED_FIELD_NAMES & set(cls.model_fields.keys())
        if reserved_conflicts:
            raise TypeError(
                f"Field name(s) {reserved_conflicts} reserved by BaseAnswer for internal use. "
                f"Please rename your field(s) to avoid collision."
            )

        verified = cls._get_verified_fields()
        if not verified:
            return

        def _has_own_method(target_cls: type, method_name: str) -> bool:
            """Check if any class in MRO (between cls and BaseAnswer) defines method."""
            for klass in target_cls.__mro__:
                if klass is BaseAnswer:
                    break
                if method_name in klass.__dict__:
                    return True
            return False

        if not _has_own_method(cls, "verify"):
            cls.verify = BaseAnswer._auto_verify  # type: ignore[attr-defined]
        if not _has_own_method(cls, "verify_granular"):
            cls.verify_granular = BaseAnswer._auto_verify_granular  # type: ignore[attr-defined]

    @classmethod
    def get_source_code(cls) -> str | None:
        """Get the source code of this Answer class.

        Returns:
            The source code string if available, None otherwise.

        For file-based classes, source code is captured automatically.
        For exec-created classes, source code must be set manually.
        """
        return cls._source_code

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Generate JSON schema with verification metadata stripped.

        Overrides Pydantic's default to ensure __verification__ metadata
        (containing ground truth values) is never exposed in the schema.
        This prevents ground truth leakage to LLM judges regardless of
        which adapter or code path generates the schema.

        All args are forwarded to Pydantic's model_json_schema().
        """
        schema = super().model_json_schema(*args, **kwargs)

        def _strip_verification(obj: Any) -> None:
            if isinstance(obj, dict):
                obj.pop("__verification__", None)
                for value in obj.values():
                    _strip_verification(value)
            elif isinstance(obj, list):
                for item in obj:
                    _strip_verification(item)

        _strip_verification(schema)
        return schema

    @classmethod
    def set_source_code_from_notebook(cls) -> None:
        """Capture source code from notebook cell history (Jupyter/IPython only).

        This is a convenience method for interactive environments where
        inspect.getsource() doesn't work. It attempts to find the class
        definition in the recent cell execution history.

        Usage in notebook:
            class Answer(BaseAnswer):
                # your class definition
                pass
            Answer.set_source_code_from_notebook()
        """
        try:
            # Try to get IPython instance (works in Jupyter notebooks)
            from IPython import get_ipython  # type: ignore[attr-defined]

            ip = get_ipython()  # type: ignore[no-untyped-call]
            if ip is None:
                print("Warning: Not in an IPython/Jupyter environment")
                return

            # Get recent cell history
            history = list(ip.history_manager.get_range())

            # Look for the class definition in recent history (last 10 cells)
            class_name = cls.__name__
            for _, _, cell_content in reversed(history[-10:]):
                if f"class {class_name}(" in cell_content:
                    # Extract just the class definition part
                    lines = cell_content.strip().split("\n")
                    class_lines = []
                    in_class = False
                    base_indent = None

                    for line in lines:
                        if f"class {class_name}(" in line:
                            in_class = True
                            base_indent = len(line) - len(line.lstrip())
                            class_lines.append(line)
                        elif in_class:
                            if line.strip() == "" or (
                                base_indent is not None and len(line) - len(line.lstrip()) > base_indent
                            ):
                                class_lines.append(line)
                            else:
                                # End of class definition
                                break

                    if class_lines:
                        cls._source_code = "\n".join(class_lines)
                        print(f"✓ Source code captured for {class_name}")
                        return

            print(f"Warning: Could not find class definition for {class_name} in recent history")

        except ImportError:
            print("Warning: IPython not available. This method only works in Jupyter notebooks.")
        except Exception as e:
            print(f"Warning: Could not capture source code: {e}")

    def model_post_init(self, __context: Any) -> None:
        """Post-init hook for auto-generating ground truth from VerifiedField metadata.

        For templates using VerifiedField, this automatically populates
        self.correct with {field_name: ground_truth} if not already set.
        Classic templates and templates with custom ground_truth() are unaffected
        because the bridge in __init_subclass__ overrides this method.
        """
        if not hasattr(self, "correct") or getattr(self, "correct", None) is None:
            verified = self.__class__._get_verified_fields()
            if verified:
                self.correct = {name: meta.ground_truth for name, meta in verified.items()}

    def set_question_id(self, question_id: str) -> None:
        """Set the question ID programmatically.

        Args:
            question_id: The unique identifier for the question this answer relates to.
        """
        self.id = question_id

    @classmethod
    def _get_verified_fields(cls) -> dict[str, Any]:
        """Extract VerificationMeta from all VerifiedField-annotated fields.

        Returns:
            Mapping of field name to VerificationMeta for fields that carry
            verification metadata in json_schema_extra["__verification__"].
        """
        from karenina.schemas.entities.verified_field import VerificationMeta

        result: dict[str, VerificationMeta] = {}
        for name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and "__verification__" in extra:
                meta = VerificationMeta.model_validate(extra["__verification__"])
                result[name] = meta
        return result

    def _clear_verification_cache(self) -> None:
        """Clear the cached _field_results, forcing recomputation on next call.

        Call this after mutating field values if you need _compute_field_results()
        to reflect the updated state.
        """
        self.__dict__.pop("_field_results", None)

    def _compute_field_results(self) -> dict[str, bool]:
        """Evaluate all VerifiedField checks and cache in _field_results.

        Results are cached in self.__dict__["_field_results"] after the first
        call. Subsequent calls return the cached value without recomputation.
        Call _clear_verification_cache() to invalidate the cache after
        mutating field values.

        For TracePrimitive fields, reads self._raw_trace and compares
        check_trace() result against bool(meta.ground_truth). For parsed
        fields, calls primitive.check(extracted, ground_truth).

        Returns:
            Mapping of field name to pass/fail boolean.

        Raises:
            ValueError: If a trace field is used but _raw_trace is not set.
        """
        cached = self.__dict__.get("_field_results")
        if cached is not None:
            return cast(dict[str, bool], cached)

        from karenina.schemas.primitives import TracePrimitive
        from karenina.schemas.primitives.registry import _reconstruct_primitive

        verified = self.__class__._get_verified_fields()
        results: dict[str, bool] = {}

        for name, meta in verified.items():
            primitive = _reconstruct_primitive(meta.verify_with)

            if isinstance(primitive, TracePrimitive):
                raw_trace = getattr(self, "_raw_trace", None)
                if raw_trace is None:
                    raise ValueError(f"Field {name!r} uses a TracePrimitive but requires _raw_trace to be set")
                trace_result = primitive.check_trace(raw_trace)
                results[name] = trace_result == bool(meta.ground_truth)
            else:
                extracted = getattr(self, name)
                results[name] = primitive.check(extracted, meta.ground_truth)

        self.__dict__["_field_results"] = results
        return results

    def _auto_verify(self) -> bool:
        """Auto-generated verify() for VerifiedField templates.

        If the subclass defines a VerificationStrategy inner class with a
        verify_strategy attribute, uses evaluate_strategy() to combine field
        results. Otherwise, requires all fields to pass.

        Returns:
            True if verification passes.

        Raises:
            NotImplementedError: If the template has no VerifiedField fields
                (classic templates must define their own verify()).
        """
        from karenina.schemas.entities.composition import evaluate_strategy

        verified = self.__class__._get_verified_fields()
        if not verified:
            raise NotImplementedError("No VerifiedField fields found. Define verify() manually for classic templates.")

        field_results = self._compute_field_results()

        # Check for VerificationStrategy inner class
        strategy_cls = getattr(self.__class__, "VerificationStrategy", None)
        if strategy_cls is not None:
            strategy = getattr(strategy_cls, "verify_strategy", None)
            if strategy is not None:
                return evaluate_strategy(strategy, field_results)

        return all(field_results.values())

    def _auto_verify_granular(self) -> float:
        """Auto-generated verify_granular() for VerifiedField templates.

        For AllOf (default, no VerificationStrategy): computes a flat weighted
        average over all VerifiedField results.

        For AnyOf: returns the max passing field weight divided by total weight.
        For AtLeastN: returns the sum of the top-N passing field weights
        divided by total weight.

        Returns:
            Score between 0.0 and 1.0.

        Raises:
            NotImplementedError: If the template has no VerifiedField fields.
        """
        verified = self.__class__._get_verified_fields()
        if not verified:
            raise NotImplementedError(
                "No VerifiedField fields found. Define verify_granular() manually for classic templates."
            )

        field_results = self._compute_field_results()

        # Check for VerificationStrategy inner class (issue 133)
        strategy_cls = getattr(self.__class__, "VerificationStrategy", None)
        strategy = getattr(strategy_cls, "verify_strategy", None) if strategy_cls else None

        if strategy is not None:
            return self._composition_aware_granular(strategy, field_results, verified)

        # Default AllOf behavior: flat weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for name, meta in verified.items():
            total_weight += meta.weight
            if field_results.get(name, False):
                weighted_sum += meta.weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _composition_aware_granular(
        strategy: Any,
        field_results: dict[str, bool],
        verified: dict[str, Any],
    ) -> float:
        """Compute granular score honoring composition strategy.

        Args:
            strategy: Composition strategy node (AllOf, AnyOf, AtLeastN).
            field_results: Per-field pass/fail booleans.
            verified: Per-field VerificationMeta (for weights).

        Returns:
            Score between 0.0 and 1.0.
        """
        from karenina.schemas.entities.composition import AnyOf, AtLeastN

        total_weight: float = sum(meta.weight for meta in verified.values())
        if total_weight == 0.0:
            return 0.0

        passing_weights: list[float] = sorted(
            [verified[name].weight for name, passed in field_results.items() if passed],
            reverse=True,
        )

        if isinstance(strategy, AnyOf):
            # AnyOf: best single passing field
            if passing_weights:
                return float(max(passing_weights)) / total_weight
            return 0.0

        if isinstance(strategy, AtLeastN):
            # AtLeastN: sum of top-N passing weights
            top_n = passing_weights[: strategy.n]
            return float(sum(top_n)) / total_weight

        # AllOf or unknown: flat weighted average
        return float(sum(passing_weights)) / total_weight

    def verify_regex(self, raw_trace: str) -> dict[str, Any]:
        """Verify regex patterns against the raw LLM response trace.

        Args:
            raw_trace: The complete raw response text from the LLM

        Returns:
            Dictionary containing regex validation results with keys:
            - 'success': bool - True if all regex patterns matched successfully
            - 'results': dict - Individual results for each regex pattern
            - 'details': dict - Detailed match information for debugging
        """
        if not hasattr(self, "regex") or not self.regex:
            return {"success": True, "results": {}, "details": {}}

        results = {}
        details = {}
        all_success = True

        for name, spec in self.regex.items():
            pattern = spec.get("pattern", "")
            expected = spec.get("expected")
            match_type = spec.get("match_type", "exact")

            try:
                result = self._verify_single_regex_pattern(raw_trace, pattern, expected, match_type)
                results[name] = result["success"]
                details[name] = result["details"]

                if not result["success"]:
                    all_success = False

            except Exception as e:
                results[name] = False
                details[name] = {"error": str(e), "pattern": pattern, "expected": expected, "match_type": match_type}
                all_success = False

        return {"success": all_success, "results": results, "details": details}

    def _verify_single_regex_pattern(self, text: str, pattern: str, expected: Any, match_type: str) -> dict[str, Any]:
        """Verify a single regex pattern against text.

        Args:
            text: Text to search in
            pattern: Regex pattern to apply
            expected: Expected result (varies by match_type)
            match_type: Type of matching - 'exact', 'contains', 'count', 'all'

        Returns:
            Dictionary with 'success' boolean and 'details' dict
        """
        matches = re.findall(pattern, text)

        details = {
            "pattern": pattern,
            "expected": expected,
            "match_type": match_type,
            "matches_found": matches,
            "match_count": len(matches),
        }

        if match_type == "exact":
            # Expected is a single string that should match exactly
            if len(matches) == 1 and matches[0] == expected:
                details["success_reason"] = "Single exact match found"
                return {"success": True, "details": details}
            else:
                details["failure_reason"] = (
                    f"Expected exactly one match of '{expected}', got {len(matches)} matches: {matches}"
                )
                return {"success": False, "details": details}

        elif match_type == "contains":
            # Expected is a string that should be found somewhere
            if expected in matches:
                details["success_reason"] = f"Expected pattern '{expected}' found in matches"
                return {"success": True, "details": details}
            else:
                details["failure_reason"] = f"Expected pattern '{expected}' not found in matches: {matches}"
                return {"success": False, "details": details}

        elif match_type == "count":
            # Expected is a number - count of matches should equal this
            if isinstance(expected, int) and len(matches) == expected:
                details["success_reason"] = f"Found exactly {expected} matches as expected"
                return {"success": True, "details": details}
            else:
                details["failure_reason"] = f"Expected {expected} matches, got {len(matches)}"
                return {"success": False, "details": details}

        elif match_type == "all":
            # Expected is a list - all items should be present in matches
            if isinstance(expected, list):
                expected_set = set(expected)
                matches_set = set(matches)
                if expected_set.issubset(matches_set):
                    details["success_reason"] = "All expected items found in matches"
                    return {"success": True, "details": details}
                else:
                    missing = expected_set - matches_set
                    details["failure_reason"] = f"Missing expected items: {list(missing)}"
                    return {"success": False, "details": details}
            else:
                details["failure_reason"] = f"Expected list for 'all' match type, got {type(expected)}"
                return {"success": False, "details": details}

        else:
            details["failure_reason"] = f"Unknown match_type: {match_type}"
            return {"success": False, "details": details}


def capture_answer_source(answer_class: type) -> type:
    """Decorator/function to automatically capture source code for Answer classes in notebooks.

    Usage as decorator:
        @capture_answer_source
        class Answer(BaseAnswer):
            # your class definition
            pass

    Usage as function:
        class Answer(BaseAnswer):
            # your class definition
            pass
        Answer = capture_answer_source(Answer)

    Args:
        answer_class: The Answer class to capture source for

    Returns:
        The same class with source code captured
    """
    if hasattr(answer_class, "set_source_code_from_notebook"):
        answer_class.set_source_code_from_notebook()
    return answer_class
