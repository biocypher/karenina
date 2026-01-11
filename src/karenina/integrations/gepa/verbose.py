"""Simple logging for GEPA optimization progress."""

import re


class SimpleLogger:
    """Lightweight logger that prints GEPA optimization progress.

    Implements GEPA's LoggerProtocol with simple print statements.

    Example:
        >>> logger = SimpleLogger()
        >>> result = gepa.optimize(..., logger=logger)
    """

    def __init__(self, show_all: bool = False):
        """Initialize the logger.

        Args:
            show_all: If True, print all GEPA messages. If False, only key events.
        """
        self.show_all = show_all
        self.baseline_score: float | None = None
        self.best_score: float = 0.0

        # Patterns for parsing GEPA messages
        # Use (\d+(?:\.\d+)?) to match floats without capturing trailing periods
        self._patterns = {
            "base_score": re.compile(r"Iteration 0: Base program.*score: (\d+(?:\.\d+)?)"),
            "selected": re.compile(r"Iteration (\d+): Selected program (\d+) score: (\d+(?:\.\d+)?)"),
            "proposed": re.compile(r"Iteration (\d+): Proposed new text for (\w+)"),
            "subsample": re.compile(
                r"Iteration (\d+): New subsample score (\d+(?:\.\d+)?) is (better|not better) than old score (\d+(?:\.\d+)?)"
            ),
            "better": re.compile(r"Iteration (\d+): Found a better program.*score (\d+(?:\.\d+)?)"),
            "val_scores": re.compile(r"Iteration (\d+): Individual valset scores.*: ({.*})"),
            "objective_scores": re.compile(r"Iteration (\d+): Objective aggregate scores.*: ({.*})"),
            "no_proposal": re.compile(r"Iteration (\d+): Reflective mutation did not propose"),
        }

    def log(self, message: str) -> None:
        """Process a GEPA log message."""
        # Base program score (iteration 0)
        if match := self._patterns["base_score"].search(message):
            score = float(match.group(1))
            self.baseline_score = score
            self.best_score = score
            print(f"[Baseline] Val score: {score:.2%}")
            return

        # Selected program for mutation
        if match := self._patterns["selected"].search(message):
            iteration = match.group(1)
            program_idx = match.group(2)
            score = float(match.group(3))
            print(f"\n[Iter {iteration}] Starting from candidate {program_idx} (score: {score:.2%})")
            return

        # Proposed new text
        if match := self._patterns["proposed"].search(message):
            iteration = match.group(1)
            target = match.group(2)
            print(f"[Iter {iteration}] Proposed new {target}")
            return

        # Subsample evaluation result
        if match := self._patterns["subsample"].search(message):
            iteration = match.group(1)
            new_score = float(match.group(2))
            is_better = match.group(3) == "better"
            old_score = float(match.group(4))
            status = "✓ better" if is_better else "✗ not better"
            print(f"[Iter {iteration}] Train sample: {new_score:.1f} vs {old_score:.1f} ({status})")
            return

        # Found better program on valset
        if match := self._patterns["better"].search(message):
            iteration = match.group(1)
            score = float(match.group(2))
            improvement = ((score - self.baseline_score) / self.baseline_score * 100) if self.baseline_score else 0
            self.best_score = max(self.best_score, score)
            print(f"[Iter {iteration}] ★ New best! Val score: {score:.2%} (+{improvement:.1f}% from baseline)")
            return

        # Per-model objective scores
        if match := self._patterns["objective_scores"].search(message):
            iteration = match.group(1)
            scores_str = match.group(2)
            # Parse the dict-like string
            try:
                # Clean up the string for display
                scores_str = scores_str.replace("'", "")
                print(f"[Iter {iteration}] Per-model scores: {scores_str}")
            except Exception:
                pass
            return

        # No proposal generated
        if match := self._patterns["no_proposal"].search(message):
            iteration = match.group(1)
            print(f"[Iter {iteration}] No new proposal generated")
            return

        # Show all other messages if requested
        if self.show_all:
            print(f"  {message}")


# Keep old names for backwards compatibility
VerboseLogger = SimpleLogger


def create_verbose_logger(
    max_iterations: int = 150,  # Ignored, kept for compatibility
    console=None,  # Ignored, kept for compatibility
    show_all_messages: bool = False,
) -> SimpleLogger:
    """Create a SimpleLogger instance.

    Args:
        max_iterations: Ignored (kept for API compatibility)
        console: Ignored (kept for API compatibility)
        show_all_messages: If True, show all raw GEPA log messages

    Returns:
        SimpleLogger instance
    """
    return SimpleLogger(show_all=show_all_messages)
