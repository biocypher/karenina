"""Utility for discovering BaseAnswer subclasses in exec() namespaces.

Replaces the hardcoded local_ns["Answer"] lookup with a scan that
finds any BaseAnswer subclass, supporting custom class names.
"""

import logging

from karenina.schemas.entities.answer import BaseAnswer

logger = logging.getLogger(__name__)


def find_answer_class(local_ns: dict[str, object]) -> type:
    """Find the leaf BaseAnswer subclass in a namespace.

    Scans for classes that inherit from BaseAnswer, excluding BaseAnswer
    itself and intermediate base classes that have further subclasses in
    the same namespace.

    Args:
        local_ns: The namespace dict (typically from exec()).

    Returns:
        The single leaf BaseAnswer subclass found.

    Raises:
        ValueError: If zero or multiple leaf subclasses are found.
    """
    all_subclasses = [
        cls
        for cls in local_ns.values()
        if isinstance(cls, type) and issubclass(cls, BaseAnswer) and cls is not BaseAnswer
    ]

    if not all_subclasses:
        raise ValueError("No BaseAnswer subclass found in template namespace")

    leaves = [
        cls
        for cls in all_subclasses
        if not any(other is not cls and issubclass(other, cls) for other in all_subclasses)
    ]

    if len(leaves) == 1:
        return leaves[0]

    names = [cls.__name__ for cls in leaves]
    raise ValueError(
        f"Multiple BaseAnswer subclasses found at leaf level: {names}. "
        "Templates must contain exactly one leaf BaseAnswer subclass."
    )
