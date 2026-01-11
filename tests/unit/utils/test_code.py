"""Unit tests for code block parsing utilities.

Tests cover:
- extract_and_combine_codeblocks() function
- Single and multiple code blocks
- Different language identifiers
- Empty and edge case inputs
- Whitespace handling
- Windows and Unix line endings
"""

import pytest

from karenina.utils.code import BACKTICK_PATTERN, extract_and_combine_codeblocks

# =============================================================================
# extract_and_combine_codeblocks() Tests
# =============================================================================


@pytest.mark.unit
def test_extract_single_codeblock() -> None:
    """Test extracting a single code block."""
    text = """Here's some code:
```python
print('hello')
```
End of text."""
    result = extract_and_combine_codeblocks(text)
    assert result == "print('hello')"


@pytest.mark.unit
def test_extract_multiple_codeblocks() -> None:
    """Test extracting and combining multiple code blocks."""
    text = """First block:
```python
print('hello')
```
Second block:
```python
print('world')
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "print('hello')\n\nprint('world')"


@pytest.mark.unit
def test_extract_codeblock_without_language() -> None:
    """Test extracting code block without language identifier."""
    text = """```
print('no language')
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "print('no language')"


@pytest.mark.unit
def test_extract_codeblock_with_various_languages() -> None:
    """Test extracting code blocks with different language identifiers."""
    text = """```javascript
console.log('js');
```

```rust
fn main() {}
```

```sql
SELECT * FROM table;
```"""
    result = extract_and_combine_codeblocks(text)
    assert "console.log('js');" in result
    assert "fn main()" in result
    assert "SELECT * FROM table" in result


@pytest.mark.unit
def test_extract_codeblock_with_language_variant() -> None:
    """Test code blocks with language variants (e.g., c++)."""
    text = """```c++
int main() { return 0; }
```"""
    result = extract_and_combine_codeblocks(text)
    assert "int main()" in result


@pytest.mark.unit
def test_no_codeblocks_returns_empty() -> None:
    """Test that text without code blocks returns empty string."""
    text = "Just regular text with no code blocks."
    result = extract_and_combine_codeblocks(text)
    assert result == ""


@pytest.mark.unit
def test_empty_string_returns_empty() -> None:
    """Test that empty string returns empty string."""
    result = extract_and_combine_codeblocks("")
    assert result == ""


@pytest.mark.unit
def test_codeblock_with_leading_trailing_whitespace() -> None:
    """Test that leading/trailing whitespace in blocks is stripped."""
    text = """```

    print('indented');

```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "print('indented');"


@pytest.mark.unit
def test_codeblock_with_internal_whitespace_preserved() -> None:
    """Test that internal whitespace is preserved."""
    text = """```
def hello():
    print('indented line')
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "def hello():\n    print('indented line')"


@pytest.mark.unit
def test_codeblock_with_spaces_after_language() -> None:
    """Test code blocks with spaces AFTER language identifier."""
    text = """```python
print('spaces')
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "print('spaces')"


@pytest.mark.unit
def test_codeblock_with_unix_line_endings() -> None:
    """Test code blocks with Unix (LF) line endings."""
    text = "```python\nprint('unix')\n```"
    result = extract_and_combine_codeblocks(text)
    assert result == "print('unix')"


@pytest.mark.unit
def test_codeblock_with_windows_line_endings() -> None:
    """Test code blocks with Windows (CRLF) line endings."""
    text = "```python\r\nprint('windows')\r\n```"
    result = extract_and_combine_codeblocks(text)
    assert result == "print('windows')"


@pytest.mark.unit
def test_codeblock_mixed_line_endings() -> None:
    """Test code blocks with mixed line endings."""
    text = "```python\r\nprint('mixed')\n```"
    result = extract_and_combine_codeblocks(text)
    assert result == "print('mixed')"


@pytest.mark.unit
def test_multiline_codeblock() -> None:
    """Test extracting multi-line code block."""
    text = """```
class Answer:
    value: str
    def verify(self):
        return True
```"""
    result = extract_and_combine_codeblocks(text)
    assert "class Answer:" in result
    assert "value: str" in result
    assert "def verify(self):" in result


@pytest.mark.unit
def test_nested_backticks_in_code() -> None:
    """Test code containing backticks (not as delimiters)."""
    text = """```python
s = "not `code`"
print(s)
```"""
    result = extract_and_combine_codeblocks(text)
    assert '"not `code`"' in result


@pytest.mark.unit
def test_codeblock_with_special_characters() -> None:
    r"""Test code block with special characters."""
    text = r"""```python
# Special chars: @#$%^&*()
regex = "\\d+"
```"""
    result = extract_and_combine_codeblocks(text)
    assert "@#$%^&*()" in result
    assert r"\d+" in result


@pytest.mark.unit
def test_codeblock_with_unicode() -> None:
    """Test code block with unicode characters."""
    text = """```python
# Emoji support 😀
message = "你好世界"
```"""
    result = extract_and_combine_codeblocks(text)
    assert "😀" in result
    assert "你好世界" in result


@pytest.mark.unit
def test_three_backticks_in_code_escaped() -> None:
    """Test code containing triple backticks is captured as is."""
    text = """```python
code_with_backticks = "```"
```"""
    result = extract_and_combine_codeblocks(text)
    # The regex is non-greedy, so it captures the first closing backticks
    assert 'code_with_backticks = "```"' in result or result.startswith("code_with_backticks")


@pytest.mark.unit
def test_consecutive_codeblocks() -> None:
    """Test consecutive code blocks with no text between."""
    text = """```
first
```
```
second
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "first\n\nsecond"


@pytest.mark.unit
def test_codeblock_with_hyphenated_language() -> None:
    """Test code block with hyphenated language identifier."""
    text = """```objective-c
NSLog(@"test");
```"""
    result = extract_and_combine_codeblocks(text)
    assert 'NSLog(@"test");' in result


@pytest.mark.unit
def test_codeblock_with_plus_language() -> None:
    """Test code block with plus sign in language."""
    text = """```c++
int x = 5;
```"""
    result = extract_and_combine_codeblocks(text)
    assert "int x = 5;" in result


@pytest.mark.unit
def test_backtick_pattern_directly() -> None:
    """Test BACKTICK_PATTERN regex directly."""
    import re

    text = """```python
code here
```"""
    matches = re.findall(BACKTICK_PATTERN, text, re.DOTALL)
    assert len(matches) == 1
    # The pattern captures everything between including the trailing newline
    assert matches[0] == "code here\n"


@pytest.mark.unit
def test_codeblock_with_empty_lines() -> None:
    """Test code block with empty lines is preserved."""
    text = """```
line1

line2
```"""
    result = extract_and_combine_codeblocks(text)
    # Empty lines within code are preserved
    assert "line1" in result
    assert "line2" in result


@pytest.mark.unit
def test_text_around_codeblocks_ignored() -> None:
    """Test that text around code blocks is not included."""
    text = """Before
```python
code
```
After"""
    result = extract_and_combine_codeblocks(text)
    assert "Before" not in result
    assert "After" not in result
    assert result == "code"


@pytest.mark.unit
def test_incomplete_backtick_partial_match() -> None:
    """Test that partial backtick sequences may still match."""
    # The non-greedy regex matches between ```python and the next ``` it finds
    text = """```python
incomplete block
Another ``` not closed"""
    result = extract_and_combine_codeblocks(text)
    # The regex finds ```python...incomplete block\nAnother ``` as a match
    assert "incomplete block" in result


@pytest.mark.unit
def test_four_backticks_gets_matched() -> None:
    """Test that four backticks still match (``` found within the sequence)."""
    # The input is: ```` followed by content followed by ````
    # The regex finds ``` starting at position 1 (the 2nd-4th backticks)
    # and matches until the closing ```
    text = """````
not a code block
````"""
    result = extract_and_combine_codeblocks(text)
    # The pattern matches starting from the 2nd backtick
    assert result == "not a code block"


@pytest.mark.unit
def test_two_backticks_not_matched() -> None:
    """Test that two backticks don't create a code block."""
    text = """``not a code block``"""
    result = extract_and_combine_codeblocks(text)
    assert result == ""


@pytest.mark.unit
def test_codeblock_with_braces() -> None:
    """Test code block with curly braces."""
    text = """```javascript
function test() {
    return {key: 'value'};
}
```"""
    result = extract_and_combine_codeblocks(text)
    assert "function test()" in result
    assert "{key: 'value'}" in result


@pytest.mark.unit
def test_very_long_codeblock() -> None:
    """Test extracting a very long code block."""
    lines = [f"line_{i}" for i in range(100)]
    code = "\n".join(lines)
    text = f"```\n{code}\n```"
    result = extract_and_combine_codeblocks(text)
    assert result == code
    assert "line_0" in result
    assert "line_99" in result


@pytest.mark.unit
def test_codeblock_with_only_whitespace() -> None:
    """Test code block containing only whitespace."""
    text = """```

```"""
    result = extract_and_combine_codeblocks(text)
    assert result == ""


@pytest.mark.unit
def test_multiple_blocks_with_different_indentation() -> None:
    """Test multiple blocks each stripped independently."""
    text = """```
    indented
```

```
        more indented
```"""
    result = extract_and_combine_codeblocks(text)
    assert result == "indented\n\nmore indented"


@pytest.mark.unit
def test_codeblock_with_dollar_sign_language() -> None:
    """Test code block with dollar sign in language (e.g., in bash examples)."""
    text = """```bash
$ echo "test"
```"""
    result = extract_and_combine_codeblocks(text)
    assert '$ echo "test"' in result
