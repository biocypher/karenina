"""mkdocs hook: pygments 2.20 / pymdown-extensions 10.16 compatibility shim.

pymdown-extensions 10.16 constructs the pygments ``HtmlFormatter`` with
``filename=None`` for code blocks that carry no title. This happens for indented
code blocks that reach ``HighlightTreeprocessor.run`` (including fenced examples
that markdown falls back to as indented code, and indented code inside
mkdocstrings-rendered docstrings) as well as language-less fences.

pygments 2.20 changed ``HtmlFormatter.__init__`` to call
``html.escape(options.get("filename", ""))`` unconditionally. When the value is
``None`` rather than an empty string this raises
``AttributeError: 'NoneType' object has no attribute 'replace'`` and aborts the
whole build.

This hook coerces a ``None`` ``filename`` to ``""`` (pygments' own default),
which produces output identical to a titleless code block. It changes no
rendering. Remove it once the pinned pygments / pymdown-extensions versions
resolve the incompatibility (for example by pinning pygments below 2.19 or
upgrading pymdown-extensions past the fix).
"""

from __future__ import annotations


def _patch() -> None:
    try:
        from pygments.formatters.html import HtmlFormatter
    except Exception:
        return

    if getattr(HtmlFormatter.__init__, "_karenina_filename_shim", False):
        return

    _orig_init = HtmlFormatter.__init__

    def __init__(self, **options):
        if options.get("filename", "") is None:
            options["filename"] = ""
        _orig_init(self, **options)

    __init__._karenina_filename_shim = True
    HtmlFormatter.__init__ = __init__


_patch()


def on_config(config, **kwargs):
    _patch()
    return config
