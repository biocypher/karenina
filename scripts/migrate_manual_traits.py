#!/usr/bin/env python3
"""
Migration script to convert ManualRubricTrait to RegexTrait in checkpoint files.

This script:
1. Loads a JSON-LD checkpoint file
2. Finds all ManualRubricTrait entries (global and question-specific)
3. Converts traits with 'pattern' to RegexTrait
4. Errors on traits with 'callable_code' (cannot auto-migrate)
5. Saves the migrated checkpoint with updated version
6. Generates a migration report

Usage:
    python scripts/migrate_manual_traits.py <input_checkpoint.jsonld> [--output <output_checkpoint.jsonld>] [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def migrate_manual_trait_to_regex(rating: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """
    Migrate a ManualRubricTrait rating to RegexTrait.

    Args:
        rating: The rating dict with additionalType = GlobalManualRubricTrait or QuestionSpecificManualRubricTrait

    Returns:
        Tuple of (migrated_rating, status_message)

    Raises:
        ValueError: If the trait cannot be auto-migrated (e.g., has callable_code)
    """
    # Extract properties
    pattern = None
    case_sensitive = True  # Default
    invert_result = False  # Default
    has_callable_code = False

    if rating.get("additionalProperty"):
        for prop in rating["additionalProperty"]:
            if prop.get("name") == "pattern":
                pattern = prop.get("value")
            elif prop.get("name") == "case_sensitive":
                case_sensitive = prop.get("value", True)
            elif prop.get("name") in ["invert", "invert_result"]:  # Handle both old and new names
                invert_result = prop.get("value", False)
            elif prop.get("name") == "callable_code":
                has_callable_code = True

    # Check if it's a callable trait (cannot auto-migrate)
    if has_callable_code:
        raise ValueError(
            f"Cannot auto-migrate trait '{rating['name']}' - it contains callable_code. "
            "CallableTrait migration requires manual intervention."
        )

    # Check if it has a pattern (required for RegexTrait)
    if pattern is None:
        raise ValueError(
            f"Cannot migrate trait '{rating['name']}' - no 'pattern' found in additionalProperty. "
            "This trait cannot be converted to RegexTrait."
        )

    # Determine new additionalType
    if rating["additionalType"] == "GlobalManualRubricTrait":
        new_type = "GlobalRegexTrait"
    else:  # QuestionSpecificManualRubricTrait
        new_type = "QuestionSpecificRegexTrait"

    # Create migrated rating
    migrated = rating.copy()
    migrated["additionalType"] = new_type

    # Update additionalProperty to use standardized field names
    migrated["additionalProperty"] = [
        {"@type": "PropertyValue", "name": "pattern", "value": pattern},
        {"@type": "PropertyValue", "name": "case_sensitive", "value": case_sensitive},
        {"@type": "PropertyValue", "name": "invert_result", "value": invert_result},
    ]

    return migrated, f"Migrated '{rating['name']}' from {rating['additionalType']} to {new_type}"


def migrate_checkpoint(
    checkpoint: dict[str, Any], dry_run: bool = False
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Migrate all ManualRubricTrait entries in a checkpoint to RegexTrait.

    Args:
        checkpoint: The checkpoint dict
        dry_run: If True, don't actually modify the checkpoint

    Returns:
        Tuple of (migrated_checkpoint, success_messages, error_messages)
    """
    success_messages = []
    error_messages = []
    migration_count = 0

    # Work on a copy if not dry run
    migrated = checkpoint if dry_run else checkpoint.copy()

    # Migrate global rubric traits
    if migrated.get("rating"):
        new_global_ratings = []
        for rating in migrated["rating"]:
            if rating.get("additionalType") in ["GlobalManualRubricTrait"]:
                try:
                    migrated_rating, msg = migrate_manual_trait_to_regex(rating)
                    new_global_ratings.append(migrated_rating)
                    success_messages.append(f"[Global] {msg}")
                    migration_count += 1
                except ValueError as e:
                    error_messages.append(f"[Global] {str(e)}")
                    new_global_ratings.append(rating)  # Keep original on error
            else:
                new_global_ratings.append(rating)

        if not dry_run:
            migrated["rating"] = new_global_ratings

    # Migrate question-specific rubric traits
    if migrated.get("dataFeedElement"):
        new_feed_elements = []
        for item in migrated["dataFeedElement"]:
            question = item.get("item", {})
            if question.get("rating"):
                new_question_ratings = []
                for rating in question["rating"]:
                    if rating.get("additionalType") in ["QuestionSpecificManualRubricTrait"]:
                        try:
                            migrated_rating, msg = migrate_manual_trait_to_regex(rating)
                            new_question_ratings.append(migrated_rating)
                            success_messages.append(f"[Question: {question.get('text', 'unknown')[:50]}...] {msg}")
                            migration_count += 1
                        except ValueError as e:
                            error_messages.append(f"[Question: {question.get('text', 'unknown')[:50]}...] {str(e)}")
                            new_question_ratings.append(rating)  # Keep original on error
                    else:
                        new_question_ratings.append(rating)

                if not dry_run:
                    question["rating"] = new_question_ratings

            new_feed_elements.append(item)

        if not dry_run:
            migrated["dataFeedElement"] = new_feed_elements

    # Update metadata
    if not dry_run and migration_count > 0:
        migrated["dateModified"] = datetime.now().isoformat()
        # Update version if present
        if migrated.get("version"):
            # Increment patch version (e.g., 1.0.0 -> 1.0.1)
            version_parts = migrated["version"].split(".")
            if len(version_parts) == 3:
                version_parts[2] = str(int(version_parts[2]) + 1)
                migrated["version"] = ".".join(version_parts)

    return migrated, success_messages, error_messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ManualRubricTrait to RegexTrait in checkpoint files")
    parser.add_argument("input", type=Path, help="Input checkpoint file (JSON-LD format)")
    parser.add_argument("--output", "-o", type=Path, help="Output checkpoint file (defaults to input_migrated.jsonld)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--report", "-r", type=Path, help="Save migration report to file")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = args.output or args.input.parent / f"{args.input.stem}_migrated{args.input.suffix}"

    # Load checkpoint
    print(f"Loading checkpoint from {args.input}...")
    try:
        with open(args.input) as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Migrate
    print(f"\n{'DRY RUN: ' if args.dry_run else ''}Migrating ManualRubricTrait → RegexTrait...")
    migrated, success_messages, error_messages = migrate_checkpoint(checkpoint, dry_run=args.dry_run)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"Migration {'Preview' if args.dry_run else 'Complete'}")
    print(f"{'=' * 80}\n")

    if success_messages:
        print(f"✅ Successfully migrated {len(success_messages)} traits:\n")
        for msg in success_messages:
            print(f"  {msg}")
    else:
        print("No ManualRubricTrait entries found to migrate.")

    if error_messages:
        print(f"\n❌ Errors ({len(error_messages)}):\n")
        for msg in error_messages:
            print(f"  {msg}")

    # Save migrated checkpoint
    if not args.dry_run and success_messages:
        print(f"\nSaving migrated checkpoint to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(migrated, f, indent=2)
        print(f"✅ Saved migrated checkpoint to {output_path}")

    # Generate report
    if args.report and success_messages:
        report = {
            "migration_date": datetime.now().isoformat(),
            "input_file": str(args.input),
            "output_file": str(output_path) if not args.dry_run else None,
            "dry_run": args.dry_run,
            "traits_migrated": len(success_messages),
            "errors": len(error_messages),
            "success_messages": success_messages,
            "error_messages": error_messages,
        }

        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Migration report saved to {args.report}")

    # Exit with error code if there were errors
    if error_messages:
        sys.exit(1)


if __name__ == "__main__":
    main()
