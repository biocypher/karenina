"""Version utilities for Karenina."""


def get_karenina_version() -> str:
    """Get the current Karenina version."""
    try:
        import karenina

        return getattr(karenina, "__version__", "unknown")
    except ImportError:
        return "unknown"
