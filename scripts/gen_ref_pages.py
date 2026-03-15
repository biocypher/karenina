"""Generate API reference pages for mkdocs from source code."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path("src")

# Modules to skip entirely
SKIP_MODULES = {"__main__", "__pycache__", "_archive"}

# Core API modules get top-level nav entries
CORE_PACKAGES = {
    "karenina",
    "karenina.benchmark",
    "karenina.schemas",
    "karenina.schemas.entities",
    "karenina.schemas.config",
    "karenina.schemas.verification",
    "karenina.schemas.results",
    "karenina.schemas.outputs",
    "karenina.schemas.dataframes",
    "karenina.schemas.checkpoint",
    "karenina.ports",
    "karenina.adapters",
    "karenina.exceptions",
}

# Internal packages go in a collapsed subsection
INTERNAL_PACKAGES = {
    "karenina.benchmark.core",
    "karenina.benchmark.verification",
    "karenina.benchmark.authoring",
    "karenina.benchmark.task_eval",
    "karenina.storage",
    "karenina.cli",
    "karenina.utils",
    "karenina.integrations",
}


def _should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    return any(part in SKIP_MODULES for part in path.parts)


def _module_path_to_dotted(module_path: Path) -> str:
    """Convert a file path to a dotted module name."""
    parts = list(module_path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_core(dotted: str) -> bool:
    """Check if a module belongs to core API."""
    if dotted in CORE_PACKAGES:
        return True
    # Check if it's a direct child of a core package
    parent = dotted.rsplit(".", 1)[0] if "." in dotted else dotted
    return parent in CORE_PACKAGES


def _nav_parts(dotted: str, is_core: bool) -> tuple[str, ...]:
    """Build nav tuple for a module."""
    # Strip the 'karenina' prefix for display
    parts = dotted.split(".")
    if is_core:
        return ("Core API", *parts)
    return ("Internals", *parts)


# Collect all Python modules
core_modules: list[tuple[str, Path]] = []
internal_modules: list[tuple[str, Path]] = []

for path in sorted(src.rglob("*.py")):
    if _should_skip(path):
        continue

    # Get module path relative to src/
    module_path = path.relative_to(src)
    dotted = _module_path_to_dotted(module_path)

    # Skip test files
    if "tests" in dotted or "test_" in path.name:
        continue

    # Skip private modules (but keep __init__)
    if path.name.startswith("_") and path.name != "__init__.py":
        continue

    is_core = _is_core(dotted)

    if is_core:
        core_modules.append((dotted, module_path))
    else:
        internal_modules.append((dotted, module_path))

# Generate pages for all modules
for dotted, module_path in core_modules + internal_modules:
    is_core = dotted in CORE_PACKAGES or _is_core(dotted)
    doc_path = Path(dotted.replace(".", "/") + ".md")
    full_doc_path = Path("reference", "api", doc_path)

    nav_parts = _nav_parts(dotted, is_core)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{dotted}`\n\n")
        fd.write(f"::: {dotted}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / module_path)

# Write the SUMMARY.md for literate-nav
with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
