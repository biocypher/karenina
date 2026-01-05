"""Benchmark splitting utilities for GEPA optimization.

Provides functions to split karenina benchmarks into train/val/test sets
for use with GEPA's optimization loop.
"""

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from karenina.integrations.gepa.data_types import BenchmarkSplit, KareninaDataInst

if TYPE_CHECKING:
    from karenina.benchmark.benchmark import Benchmark


def split_benchmark(
    benchmark: "Benchmark",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float | None = None,
    seed: int | None = None,
    stratify_by: str | None = None,
) -> BenchmarkSplit:
    """Split benchmark questions into train/val (default) or train/val/test sets.

    Default behavior creates an 80/20 train/val split. Optionally include
    a test set by specifying test_ratio.

    Args:
        benchmark: Karenina Benchmark object to split
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.2)
        test_ratio: Optional fraction for testing. If None, no test set created.
        seed: Random seed for reproducibility
        stratify_by: Attribute to stratify by (e.g., "author", "difficulty").
                     Preserves distribution of this attribute across splits.

    Returns:
        BenchmarkSplit with train, val, and optionally test sets.

    Raises:
        ValueError: If ratios don't sum to 1.0 or benchmark has no questions.

    Examples:
        >>> # Default 80/20 split
        >>> split = split_benchmark(benchmark)

        >>> # 70/15/15 split with test set
        >>> split = split_benchmark(benchmark, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        >>> # Stratified by author
        >>> split = split_benchmark(benchmark, stratify_by="author")
    """
    # Validate ratios
    if test_ratio is not None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total}")
    else:
        total = train_ratio + val_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio must equal 1.0 when test_ratio is None, got {total}")

    # Get all question IDs
    question_ids = benchmark.get_question_ids()
    if not question_ids:
        raise ValueError("Benchmark has no questions to split")

    # Convert to data instances
    all_insts = questions_to_data_insts(benchmark, question_ids)

    # Set random seed
    if seed is not None:
        random.seed(seed)

    if stratify_by:
        # Stratified split
        return _stratified_split(
            all_insts,
            stratify_by=stratify_by,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        # Simple random split
        return _random_split(
            all_insts,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )


def _random_split(
    instances: list[KareninaDataInst],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None,
    seed: int | None,
) -> BenchmarkSplit:
    """Perform random split."""
    # Shuffle
    shuffled = instances.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:] if test_ratio is not None else None

    # Ensure no empty sets
    if not train:
        raise ValueError("Training set is empty. Check ratios and benchmark size.")
    if not val:
        raise ValueError("Validation set is empty. Check ratios and benchmark size.")

    return BenchmarkSplit(train=train, val=val, test=test, seed=seed)


def _stratified_split(
    instances: list[KareninaDataInst],
    stratify_by: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None,
    seed: int | None,
) -> BenchmarkSplit:
    """Perform stratified split preserving distribution of stratify_by attribute."""
    # Group by stratification attribute
    groups: dict[Any, list[KareninaDataInst]] = defaultdict(list)
    for inst in instances:
        key = inst.metadata.get(stratify_by, "unknown")
        groups[key].append(inst)

    train: list[KareninaDataInst] = []
    val: list[KareninaDataInst] = []
    test: list[KareninaDataInst] | None = [] if test_ratio is not None else None

    # Split each group proportionally
    for group_insts in groups.values():
        random.shuffle(group_insts)
        n = len(group_insts)
        train_end = max(1, int(n * train_ratio))  # At least 1 in train
        val_end = train_end + max(1, int(n * val_ratio))  # At least 1 in val

        train.extend(group_insts[:train_end])
        val.extend(group_insts[train_end:val_end])
        if test is not None:
            test.extend(group_insts[val_end:])

    # Shuffle final sets
    random.shuffle(train)
    random.shuffle(val)
    if test is not None:
        random.shuffle(test)

    return BenchmarkSplit(train=train, val=val, test=test, seed=seed)


def split_by_attribute(
    benchmark: "Benchmark",
    attribute: str,
    train_values: list[str],
    val_values: list[str],
    test_values: list[str] | None = None,
) -> BenchmarkSplit:
    """Split benchmark by specific attribute values.

    Questions are assigned to sets based on the value of a metadata attribute.
    This is useful for held-out evaluation (e.g., train on some authors,
    validate on others).

    Args:
        benchmark: Karenina Benchmark object
        attribute: Metadata attribute to split by (e.g., "author", "source")
        train_values: Attribute values that go to training set
        val_values: Attribute values that go to validation set
        test_values: Optional attribute values for test set

    Returns:
        BenchmarkSplit with train, val, and optionally test sets.

    Raises:
        ValueError: If any set is empty or values overlap.

    Example:
        >>> split = split_by_attribute(
        ...     benchmark,
        ...     attribute="author",
        ...     train_values=["alice", "bob"],
        ...     val_values=["charlie"],
        ...     test_values=["diana"]
        ... )
    """
    # Check for overlapping values
    all_values = set(train_values) | set(val_values)
    if test_values:
        all_values |= set(test_values)

    if len(all_values) < len(train_values) + len(val_values) + (len(test_values) if test_values else 0):
        raise ValueError("Attribute values must not overlap between sets")

    # Get all question IDs
    question_ids = benchmark.get_question_ids()
    all_insts = questions_to_data_insts(benchmark, question_ids)

    # Assign to sets
    train_set = set(train_values)
    val_set = set(val_values)
    test_set = set(test_values) if test_values else set()

    train: list[KareninaDataInst] = []
    val: list[KareninaDataInst] = []
    test: list[KareninaDataInst] | None = [] if test_values else None

    for inst in all_insts:
        value = inst.metadata.get(attribute, "unknown")
        if value in train_set:
            train.append(inst)
        elif value in val_set:
            val.append(inst)
        elif value in test_set and test is not None:
            test.append(inst)
        # Instances with values not in any set are excluded

    if not train:
        raise ValueError(f"No questions found with {attribute} in {train_values}")
    if not val:
        raise ValueError(f"No questions found with {attribute} in {val_values}")

    return BenchmarkSplit(train=train, val=val, test=test)


def questions_to_data_insts(
    benchmark: "Benchmark",
    question_ids: list[str],
) -> list[KareninaDataInst]:
    """Convert benchmark questions to GEPA data instances.

    Args:
        benchmark: Karenina Benchmark object
        question_ids: List of question IDs to convert

    Returns:
        List of KareninaDataInst objects ready for GEPA.
    """
    instances: list[KareninaDataInst] = []

    for qid in question_ids:
        question = benchmark.get_question(qid)
        if question is None:
            continue

        # Get template code
        template_code = benchmark.get_template(qid)

        # Get rubric if available
        rubric = None
        if hasattr(benchmark, "get_rubric"):
            rubric_obj = benchmark.get_rubric(qid)
            if rubric_obj:
                rubric = rubric_obj.to_dict() if hasattr(rubric_obj, "to_dict") else {}

        # Get few-shot examples (question is a dict from get_question)
        few_shot = question.get("few_shot_examples")

        # Build metadata from question attributes
        metadata: dict[str, Any] = {}
        if "author" in question:
            metadata["author"] = question["author"]
        if "tags" in question or "keywords" in question:
            metadata["tags"] = question.get("tags") or question.get("keywords")
        if "finished" in question:
            metadata["finished"] = question["finished"]

        instances.append(
            KareninaDataInst(
                question_id=qid,
                question_text=question.get("question", ""),
                raw_answer=question.get("raw_answer", ""),
                template_code=template_code or "",
                rubric=rubric,
                few_shot_examples=few_shot,
                metadata=metadata,
            )
        )

    return instances
