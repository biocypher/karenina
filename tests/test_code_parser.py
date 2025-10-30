from karenina.utils.code import extract_and_combine_codeblocks


def test_extract_and_combine_codeblocks_empty() -> None:
    """Test extracting code blocks from empty text."""
    result = extract_and_combine_codeblocks("")
    assert result == ""


def test_extract_and_combine_codeblocks_no_blocks() -> None:
    """Test extracting code blocks from text with no code blocks."""
    text = "This is some text without any code blocks."
    result = extract_and_combine_codeblocks(text)
    assert result == ""


def test_extract_and_combine_codeblocks_single_block() -> None:
    """Test extracting a single code block."""
    text = """
    Here's some code:
    ```python
    def hello():
        print('world')
    ```
    """
    result = extract_and_combine_codeblocks(text)
    assert "def hello():" in result
    assert "print('world')" in result
    assert "```" not in result


def test_extract_and_combine_codeblocks_multiple_blocks() -> None:
    """Test extracting multiple code blocks."""
    text = """
    First block:
    ```python
    def hello():
        print('world')
    ```
    Second block:
    ```
    def goodbye():
        print('earth')
    ```
    """
    result = extract_and_combine_codeblocks(text)
    assert "def hello():" in result
    assert "print('world')" in result
    assert "def goodbye():" in result
    assert "print('earth')" in result
    assert "```" not in result


def test_extract_and_combine_codeblocks_with_language_identifiers() -> None:
    """Test extracting code blocks with language identifiers."""
    text = """
    ```python
    def hello():
        print('world')
    ```
    ```javascript
    function hello() {
        console.log('world');
    }
    ```
    """
    result = extract_and_combine_codeblocks(text)
    assert "def hello():" in result
    assert "print('world')" in result
    assert "function hello()" in result
    assert "console.log('world')" in result
    assert "python" not in result
    assert "javascript" not in result
    assert "```" not in result


def test_extract_and_combine_codeblocks_with_whitespace() -> None:
    """Test extracting code blocks with various whitespace patterns."""
    text = """
    ```python
        def hello():
            print('world')
    ```
    ```
    def goodbye():
        print('earth')
    ```
    """
    result = extract_and_combine_codeblocks(text)
    assert "def hello():" in result
    assert "print('world')" in result
    assert "def goodbye():" in result
    assert "print('earth')" in result
    assert "```" not in result


def test_extract_and_combine_codeblocks_with_trailing_whitespace_after_language() -> None:
    """Test extracting code blocks with trailing whitespace after the language identifier."""
    text = """
    ```python
    def foo():
        return 42
    ```
    """
    result = extract_and_combine_codeblocks(text)
    assert "def foo():" in result
    assert "return 42" in result
    assert "```" not in result


def test_extract_and_combine_codeblocks_windows_line_endings() -> None:
    """Test extracting code blocks with Windows (CRLF) line endings."""
    text = """\r\n```python\r\ndef bar():\r\n    return 24\r\n```\r\n"""
    result = extract_and_combine_codeblocks(text)
    assert "def bar():" in result
    assert "return 24" in result
    assert "```" not in result
