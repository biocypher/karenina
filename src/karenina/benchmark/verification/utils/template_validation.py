"""Template validation logic for verification system."""

import inspect
from typing import Any

from karenina.schemas.entities import BaseAnswer

from .class_discovery import find_answer_class


def _build_exec_namespace() -> dict[str, Any]:
    """Build a namespace dict for exec() of template code.

    Includes BaseAnswer, Pydantic Field, typing utilities, and all
    VerifiedField types (primitives, composition nodes, normalizers).

    Returns:
        Namespace dict suitable as global_ns for exec().
    """
    from typing import Literal, Optional, Union

    from pydantic import Field

    from karenina.schemas.entities import (
        AllOf,
        AnyOf,
        AtLeastN,
        BooleanMatch,
        ConditionalGroundTruth,
        ContainsAll,
        ContainsAny,
        DateMatch,
        DateRange,
        DateTolerance,
        ExactMatch,
        FieldCheck,
        GroundTruthCase,
        LiteralMatch,
        NumericExact,
        NumericGraded,
        NumericMaximum,
        NumericMinimum,
        NumericRange,
        NumericRangeGraded,
        NumericThresholdGraded,
        NumericTolerance,
        OrderedMatch,
        RegexMatch,
        SemanticMatch,
        SetContainment,
        SynonymMap,
        TraceContains,
        TraceLength,
        TraceRegex,
        VerifiedField,
    )

    return {
        "__builtins__": __builtins__,
        # Core
        "BaseAnswer": BaseAnswer,
        "Field": Field,
        # Typing
        "List": list,
        "Dict": dict,
        "Optional": Optional,
        "Union": Union,
        "Any": Any,
        "Literal": Literal,
        # VerifiedField system
        "VerifiedField": VerifiedField,
        # Primitives
        "BooleanMatch": BooleanMatch,
        "ExactMatch": ExactMatch,
        "ContainsAny": ContainsAny,
        "ContainsAll": ContainsAll,
        "RegexMatch": RegexMatch,
        "SemanticMatch": SemanticMatch,
        "NumericExact": NumericExact,
        "NumericTolerance": NumericTolerance,
        "NumericGraded": NumericGraded,
        "NumericRangeGraded": NumericRangeGraded,
        "NumericThresholdGraded": NumericThresholdGraded,
        "NumericMinimum": NumericMinimum,
        "NumericMaximum": NumericMaximum,
        "NumericRange": NumericRange,
        "SetContainment": SetContainment,
        "OrderedMatch": OrderedMatch,
        "LiteralMatch": LiteralMatch,
        "DateMatch": DateMatch,
        "DateTolerance": DateTolerance,
        "DateRange": DateRange,
        "TraceRegex": TraceRegex,
        "TraceContains": TraceContains,
        "TraceLength": TraceLength,
        # Composition
        "AllOf": AllOf,
        "AnyOf": AnyOf,
        "AtLeastN": AtLeastN,
        "FieldCheck": FieldCheck,
        # Normalizers
        "SynonymMap": SynonymMap,
        # Conditional ground truth
        "ConditionalGroundTruth": ConditionalGroundTruth,
        "GroundTruthCase": GroundTruthCase,
    }


def validate_answer_template(template_code: str) -> tuple[bool, str | None, type | None]:
    """Validate that template code defines a proper Answer class.

    Discovers the answer class by scanning for the leaf BaseAnswer subclass,
    supporting custom class names (not just "Answer").

    Args:
        template_code: Python source code defining a BaseAnswer subclass.

    Returns:
        Tuple of (is_valid, error_message, Answer_class).
    """
    try:
        global_ns = _build_exec_namespace()
        local_ns: dict[str, Any] = {}

        exec(template_code, global_ns, local_ns)

        # Discover the answer class (supports custom names)
        try:
            Answer = find_answer_class(local_ns)
        except ValueError as e:
            return False, str(e), None

        # Store the template code for exec-created classes
        # (since inspect.getsource() won't work for them)
        Answer._source_code = template_code  # type: ignore[attr-defined]

        if not inspect.isclass(Answer):
            return False, "Answer is not a class", None

        if not issubclass(Answer, BaseAnswer):
            return False, "Answer class must inherit from BaseAnswer", None

        # Check if it has a verify method (not required for regex-only or VerifiedField templates)
        from .template_parsing_helpers import is_regex_only_template

        has_verified_fields = bool(Answer._get_verified_fields())
        if not is_regex_only_template(Answer) and not has_verified_fields:
            if not hasattr(Answer, "verify"):
                return False, "does not have a 'verify' method", None
            if not callable(getattr(Answer, "verify", None)):
                return False, "verify must be a callable method", None

        # The 'correct' field is optional, but if present via ground_truth/model_post_init, it must be a dict
        has_init = "model_post_init" in Answer.__dict__ or "ground_truth" in Answer.__dict__
        if has_init:
            try:
                from .template_parsing_helpers import create_test_instance_from_answer_class

                test_instance, ground_truth = create_test_instance_from_answer_class(Answer)
                if ground_truth is not None and not isinstance(ground_truth, dict):
                    return False, "ground_truth/model_post_init must assign 'self.correct' as a dictionary", None
            except Exception as e:
                return False, f"Error testing ground_truth/model_post_init: {e}", None

        return True, None, Answer

    except SyntaxError as e:
        return False, f"Error executing template code: {e}", None
    except Exception as e:
        return False, f"Error executing template code: {e}", None
