# LLM Interface

The `karenina.llm.interface` module provides unified access to multiple LLM providers through a session-based conversation system.

## Core Classes

### ChatSession

Manages conversation state and LLM interactions.

**Class Definition:**
```python
class ChatSession:
    def __init__(self, session_id: str, model: str, provider: str, temperature: float = 0.7):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.messages: List = []
        self.llm = None
        self.created_at = datetime.now()
        self.last_used = datetime.now()

    def initialize_llm(self):
        """Initialize the LLM if not already done."""

    def add_message(self, message, is_human: bool = True):
        """Add a message to the conversation history."""

    def add_system_message(self, message: str):
        """Add a system message to the conversation."""
```

**Parameters:**
- `session_id` (str): Unique identifier for the conversation
- `model` (str): Model name (e.g., "gpt-4", "gemini-2.0-flash")
- `provider` (str): Provider identifier ("openai", "google_genai", "anthropic")
- `temperature` (float): Sampling temperature [0.0-1.0]

**Attributes:**
- `messages`: List of conversation messages (LangChain format)
- `llm`: Initialized LLM instance (lazy-loaded)
- `created_at`: Session creation timestamp
- `last_used`: Last interaction timestamp

### ChatRequest

Pydantic model for API requests.

**Class Definition:**
```python
class ChatRequest(BaseModel):
    model: str
    provider: str
    message: str
    session_id: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = 0.7
```

### ChatResponse

Pydantic model for API responses.

**Class Definition:**
```python
class ChatResponse(BaseModel):
    session_id: str
    message: str
    model: str
    provider: str
    timestamp: str
```

## Core Functions

### init_chat_model_unified

Unified LLM initialization across providers.

**Function Signature:**
```python
def init_chat_model_unified(model: str, provider: str = None, interface: str = "langchain", **kwargs) -> ChatSession:
    """Initialize a chat model using the unified interface.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4", "claude-3-sonnet")
        provider: The model provider (e.g., "google_genai", "openai", "anthropic")
        interface: The interface to use for model initialization ("langchain", "openrouter")
        **kwargs: Additional keyword arguments passed to model initialization

    Returns:
        ChatSession: An initialized chat model instance ready for inference

    Raises:
        ValueError: If an unsupported interface is specified
    """
```

**Usage Examples:**

```python
# Google Gemini via LangChain
llm = init_chat_model_unified("gemini-2.0-flash", "google_genai")

# OpenAI via OpenRouter
llm = init_chat_model_unified("gpt-4", interface="openrouter")

# Custom temperature
llm = init_chat_model_unified("claude-3-sonnet", "anthropic", temperature=0.2)
```

### call_model

Primary interface for LLM interactions with session management.

**Function Signature:**
```python
def call_model(
    model: str,
    provider: str,
    message: str,
    session_id: Optional[str] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.7,
) -> ChatResponse:
    """Call a language model and return the response, supporting conversational context.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4")
        provider: The model provider (e.g., "google_genai", "openai")
        message: The user message to send
        session_id: Optional session ID for continuing a conversation
        system_message: Optional system message to set context
        temperature: Model temperature for response generation

    Returns:
        ChatResponse with the model's response and session information

    Raises:
        LLMNotAvailableError: If LangChain is not available
        SessionError: If there's an error with session management
        LLMError: For other LLM-related errors
    """
```

**Parameters:**
- `model` (str): Model identifier
- `provider` (str): Provider name
- `message` (str): User message
- `session_id` (Optional[str]): Existing session ID or None for new session
- `system_message` (Optional[str]): System prompt
- `temperature` (float): Sampling temperature

**Returns:**
- `ChatResponse`: Response object with message content and metadata

**Usage Example:**

```python
# Single interaction
response = call_model(
    model="gpt-4",
    provider="openai",
    message="What is the capital of France?",
    temperature=0.3
)
print(response.message)  # "Paris"

# Continuing conversation
response2 = call_model(
    model="gpt-4",
    provider="openai",
    message="What's its population?",
    session_id=response.session_id
)
```

## Session Management

### get_session

Retrieve active session by ID.

::: karenina.llm.interface.get_session

### list_sessions

List all active sessions with metadata.

::: karenina.llm.interface.list_sessions

**Returns:**
```python
[
    {
        "session_id": "uuid-string",
        "model": "gpt-4",
        "provider": "openai",
        "created_at": "2024-01-01T12:00:00",
        "last_used": "2024-01-01T12:05:00",
        "message_count": 4
    }
]
```

### delete_session

Remove session from memory.

::: karenina.llm.interface.delete_session

### clear_all_sessions

Clear all active sessions.

::: karenina.llm.interface.clear_all_sessions

## Provider Implementations

### ChatOpenRouter

OpenRouter-specific LangChain implementation.

::: karenina.llm.interface.ChatOpenRouter

**Configuration:**
- Base URL: `https://openrouter.ai/api/v1`
- API Key: `OPENROUTER_API_KEY` environment variable
- Compatible with OpenAI API format

## Exception Hierarchy

### LLMError

Base exception for all LLM-related errors.

::: karenina.llm.interface.LLMError

### LLMNotAvailableError

Raised when LangChain dependencies are missing.

::: karenina.llm.interface.LLMNotAvailableError

### SessionError

Session management and validation errors.

::: karenina.llm.interface.SessionError

## Environment Configuration

Required environment variables by provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Google AI
export GOOGLE_API_KEY="AI..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
```

## Advanced Usage

### Custom System Messages

```python
response = call_model(
    model="claude-3-sonnet",
    provider="anthropic",
    message="Analyze this code",
    system_message="You are a code reviewer. Focus on security and performance."
)
```

### Session Persistence

```python
# Start conversation
response1 = call_model("gpt-4", "openai", "Hello")
session_id = response1.session_id

# Continue with context
response2 = call_model("gpt-4", "openai", "What did I just say?", session_id)

# Session automatically maintains message history
```

### Error Handling

```python
try:
    response = call_model("invalid-model", "openai", "test")
except LLMNotAvailableError:
    print("LangChain not installed")
except SessionError as e:
    print(f"Session error: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
```
