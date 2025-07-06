# LLM Providers Guide

This guide covers configuration and usage of different LLM providers through Karenina's unified interface.

## Overview

Karenina provides a unified interface for multiple LLM providers through LangChain, enabling consistent access to OpenAI, Google AI, Anthropic, and OpenRouter APIs.

## Provider Configuration

### Environment Setup

```bash
# OpenAI
export OPENAI_API_KEY="sk-your-openai-key"

# Google AI
export GOOGLE_API_KEY="AIza-your-google-key"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

# OpenRouter
export OPENROUTER_API_KEY="sk-or-your-openrouter-key"
```

### Using .env Files

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=AIza-your-google-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
OPENROUTER_API_KEY=sk-or-your-openrouter-key
```

```python
from dotenv import load_dotenv
load_dotenv()  # Automatically loaded by karenina.llm.interface
```

## OpenAI Configuration

### Supported Models

```python
from karenina.llm.interface import init_chat_model_unified, call_model

# GPT-4 family
gpt4_models = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview"
]

# GPT-3.5 family
gpt35_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k"
]

# Usage examples
for model in gpt4_models:
    llm = init_chat_model_unified(
        model=model,
        provider="openai",
        temperature=0.3
    )
    print(f"Initialized {model}")
```

### OpenAI-specific Parameters

```python
# Advanced OpenAI configuration
response = call_model(
    model="gpt-4",
    provider="openai",
    message="Explain quantum computing",
    temperature=0.7,          # Creativity control
    max_tokens=500,           # Response length limit
    top_p=0.9,               # Nucleus sampling
    frequency_penalty=0.1,    # Reduce repetition
    presence_penalty=0.1      # Encourage topic diversity
)
```

### OpenAI Usage Examples

```python
# Basic usage
response = call_model(
    model="gpt-4",
    provider="openai",
    message="What is machine learning?"
)
print(response.message)

# With system message
response = call_model(
    model="gpt-3.5-turbo",
    provider="openai",
    message="Explain this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    system_message="You are a programming tutor. Explain code clearly and concisely."
)

# Conversational usage
session_id = None
for user_message in ["Hello", "What's 2+2?", "What about 3+3?"]:
    response = call_model(
        model="gpt-3.5-turbo",
        provider="openai",
        message=user_message,
        session_id=session_id
    )
    session_id = response.session_id
    print(f"User: {user_message}")
    print(f"Assistant: {response.message}\n")
```

## Google AI Configuration

### Supported Models

```python
# Gemini models
gemini_models = [
    "gemini-2.0-flash",
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]

# Usage
for model in gemini_models:
    try:
        llm = init_chat_model_unified(
            model=model,
            provider="google_genai",
            temperature=0.2
        )
        print(f"Successfully initialized {model}")
    except Exception as e:
        print(f"Failed to initialize {model}: {e}")
```

### Google AI Usage Examples

```python
# Gemini 2.0 Flash (fast and efficient)
response = call_model(
    model="gemini-2.0-flash",
    provider="google_genai",
    message="Summarize the key points of machine learning",
    temperature=0.1  # Low temperature for factual content
)

# Gemini Pro (more capable)
response = call_model(
    model="gemini-pro",
    provider="google_genai",
    message="Write a detailed analysis of climate change impacts",
    temperature=0.5
)

# With custom system prompt
response = call_model(
    model="gemini-1.5-pro",
    provider="google_genai",
    message="Analyze this data: [1, 5, 3, 9, 2, 7]",
    system_message="You are a data analyst. Provide statistical insights."
)
```

### Google AI Best Practices

```python
# Optimal settings for different use cases
configurations = {
    "factual_qa": {
        "model": "gemini-2.0-flash",
        "temperature": 0.0,
        "system_message": "Provide accurate, factual answers."
    },
    "creative_writing": {
        "model": "gemini-pro",
        "temperature": 0.8,
        "system_message": "Be creative and engaging in your responses."
    },
    "code_analysis": {
        "model": "gemini-1.5-pro",
        "temperature": 0.2,
        "system_message": "Analyze code for bugs, performance, and best practices."
    }
}

# Use appropriate configuration
use_case = "factual_qa"
config = configurations[use_case]

response = call_model(
    model=config["model"],
    provider="google_genai",
    message="What is the boiling point of water?",
    temperature=config["temperature"],
    system_message=config["system_message"]
)
```

## Anthropic Configuration

### Supported Models

```python
# Claude models
claude_models = [
    "claude-3-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
    "claude-3.5-sonnet",
    "claude-2.1",
    "claude-2"
]

# Model capabilities comparison
model_comparison = {
    "claude-3-opus": "Highest capability, best for complex reasoning",
    "claude-3-sonnet": "Balanced performance and speed",
    "claude-3-haiku": "Fastest, good for simple tasks",
    "claude-3.5-sonnet": "Enhanced version of sonnet"
}

for model, description in model_comparison.items():
    print(f"{model}: {description}")
```

### Anthropic Usage Examples

```python
# Claude 3 Sonnet (recommended for most use cases)
response = call_model(
    model="claude-3-sonnet",
    provider="anthropic",
    message="Explain the concept of recursion in programming",
    temperature=0.3
)

# Claude 3 Opus (for complex reasoning)
response = call_model(
    model="claude-3-opus",
    provider="anthropic",
    message="Analyze the philosophical implications of artificial intelligence",
    temperature=0.5,
    system_message="You are a philosophy professor. Provide deep, nuanced analysis."
)

# Claude 3 Haiku (for quick responses)
response = call_model(
    model="claude-3-haiku",
    provider="anthropic",
    message="Define photosynthesis",
    temperature=0.1
)
```

### Anthropic Best Practices

```python
# Long-form content generation
def generate_essay(topic: str, length: str = "medium"):
    """Generate essay using Claude's writing capabilities."""

    system_prompts = {
        "short": "Write a concise 200-word essay.",
        "medium": "Write a well-structured 500-word essay.",
        "long": "Write a comprehensive 1000-word essay."
    }

    response = call_model(
        model="claude-3-sonnet",
        provider="anthropic",
        message=f"Write an essay about: {topic}",
        system_message=system_prompts.get(length, system_prompts["medium"]),
        temperature=0.7
    )

    return response.message

# Code review with Claude
def code_review(code: str):
    """Use Claude for comprehensive code review."""

    response = call_model(
        model="claude-3.5-sonnet",
        provider="anthropic",
        message=f"Review this code:\n\n```python\n{code}\n```",
        system_message="""You are an expert code reviewer. Analyze the code for:
        1. Bugs and potential issues
        2. Performance optimizations
        3. Code style and best practices
        4. Security considerations
        5. Suggestions for improvement""",
        temperature=0.2
    )

    return response.message

# Usage
essay = generate_essay("The impact of renewable energy")
review = code_review("def bubble_sort(arr): ...")
```

## OpenRouter Configuration

### Access Multiple Providers

```python
from karenina.llm.interface import ChatOpenRouter

# OpenRouter provides access to multiple models through one API
openrouter_models = [
    "anthropic/claude-3-sonnet",
    "openai/gpt-4",
    "google/gemini-pro",
    "meta-llama/llama-2-70b",
    "microsoft/wizardlm-70b",
    "nousresearch/nous-hermes-llama2-13b"
]

# Usage with OpenRouter
response = call_model(
    model="anthropic/claude-3-sonnet",
    provider="openrouter",  # Note: provider not required for OpenRouter
    message="Compare Python and JavaScript",
    interface="openrouter"  # Use OpenRouter interface
)
```

### OpenRouter-specific Features

```python
# Model comparison through OpenRouter
def compare_models_via_openrouter(question: str, models: list):
    """Compare responses from multiple models via OpenRouter."""

    results = {}

    for model in models:
        try:
            response = call_model(
                model=model,
                message=question,
                interface="openrouter",
                temperature=0.3
            )
            results[model] = response.message

        except Exception as e:
            print(f"Error with {model}: {e}")
            results[model] = None

    return results

# Usage
models_to_compare = [
    "anthropic/claude-3-sonnet",
    "openai/gpt-4",
    "google/gemini-pro"
]

question = "Explain the difference between AI and machine learning"
comparison = compare_models_via_openrouter(question, models_to_compare)

for model, response in comparison.items():
    print(f"\n{model}:")
    print(response[:200] + "..." if response and len(response) > 200 else response)
```

## Provider-Specific Optimization

### Performance Considerations

```python
# Provider performance characteristics
provider_profiles = {
    "openai": {
        "strengths": ["Consistent quality", "Wide model range", "Good documentation"],
        "best_for": ["General purpose", "Code generation", "Conversational AI"],
        "latency": "Medium",
        "cost": "Medium"
    },
    "google_genai": {
        "strengths": ["Fast inference", "Multimodal capabilities", "Cost-effective"],
        "best_for": ["Quick responses", "Factual QA", "Content generation"],
        "latency": "Low",
        "cost": "Low"
    },
    "anthropic": {
        "strengths": ["Safety focused", "Long context", "Reasoning ability"],
        "best_for": ["Complex analysis", "Safety-critical apps", "Long documents"],
        "latency": "Medium-High",
        "cost": "Medium-High"
    }
}

def select_optimal_provider(task_type: str, priority: str):
    """Select optimal provider based on task and priorities."""

    recommendations = {
        "factual_qa": {
            "speed": ("google_genai", "gemini-2.0-flash"),
            "quality": ("anthropic", "claude-3-sonnet"),
            "cost": ("google_genai", "gemini-2.0-flash")
        },
        "code_generation": {
            "speed": ("openai", "gpt-3.5-turbo"),
            "quality": ("openai", "gpt-4"),
            "cost": ("google_genai", "gemini-pro")
        },
        "creative_writing": {
            "speed": ("google_genai", "gemini-pro"),
            "quality": ("anthropic", "claude-3-opus"),
            "cost": ("google_genai", "gemini-pro")
        }
    }

    if task_type in recommendations and priority in recommendations[task_type]:
        return recommendations[task_type][priority]

    return ("openai", "gpt-3.5-turbo")  # Default

# Usage
provider, model = select_optimal_provider("factual_qa", "speed")
print(f"Recommended: {provider} with {model}")
```

### Error Handling by Provider

```python
def robust_provider_call(model: str, provider: str, message: str, fallback_providers: list = None):
    """Call LLM with fallback providers for reliability."""

    if fallback_providers is None:
        fallback_providers = ["openai", "google_genai", "anthropic"]

    providers_to_try = [provider] + [p for p in fallback_providers if p != provider]

    for attempt_provider in providers_to_try:
        try:
            response = call_model(
                model=model,
                provider=attempt_provider,
                message=message,
                temperature=0.3
            )

            if response.message.strip():  # Successful response
                return response, attempt_provider

        except Exception as e:
            print(f"Provider {attempt_provider} failed: {e}")
            continue

    raise Exception("All providers failed")

# Usage with fallback
try:
    response, used_provider = robust_provider_call(
        model="gpt-4",
        provider="openai",
        message="What is quantum computing?",
        fallback_providers=["google_genai", "anthropic"]
    )
    print(f"Success with {used_provider}: {response.message}")

except Exception as e:
    print(f"All providers failed: {e}")
```

## Session Management Across Providers

### Cross-Provider Sessions

```python
from karenina.llm.interface import list_sessions, get_session, delete_session

def manage_provider_sessions():
    """Demonstrate session management across providers."""

    # Start sessions with different providers
    providers_config = [
        ("gpt-3.5-turbo", "openai"),
        ("gemini-2.0-flash", "google_genai"),
        ("claude-3-haiku", "anthropic")
    ]

    session_ids = []

    for model, provider in providers_config:
        response = call_model(
            model=model,
            provider=provider,
            message="Hello, I'm starting a conversation",
            system_message=f"You are powered by {model} from {provider}"
        )
        session_ids.append(response.session_id)
        print(f"Started session {response.session_id} with {provider}:{model}")

    # List all active sessions
    sessions = list_sessions()
    print(f"\nActive sessions: {len(sessions)}")
    for session in sessions:
        print(f"  {session['session_id']}: {session['provider']}:{session['model']}")

    # Continue conversations
    for session_id in session_ids:
        session = get_session(session_id)
        if session:
            response = call_model(
                model=session.model,
                provider=session.provider,
                message="What's your model name?",
                session_id=session_id
            )
            print(f"\n{session.provider}:{session.model} says: {response.message}")

    # Clean up sessions
    for session_id in session_ids:
        delete_session(session_id)
        print(f"Deleted session {session_id}")

# Usage
manage_provider_sessions()
```

### Provider-Specific Session Configuration

```python
def create_specialized_sessions():
    """Create sessions optimized for different providers."""

    session_configs = {
        "openai_coding": {
            "model": "gpt-4",
            "provider": "openai",
            "system_message": "You are an expert programmer. Provide clean, efficient code.",
            "temperature": 0.2
        },
        "google_factual": {
            "model": "gemini-2.0-flash",
            "provider": "google_genai",
            "system_message": "Provide accurate, concise factual information.",
            "temperature": 0.0
        },
        "anthropic_analysis": {
            "model": "claude-3-sonnet",
            "provider": "anthropic",
            "system_message": "Provide deep, thoughtful analysis of complex topics.",
            "temperature": 0.4
        }
    }

    sessions = {}

    for session_name, config in session_configs.items():
        response = call_model(
            model=config["model"],
            provider=config["provider"],
            message="Ready to help!",
            system_message=config["system_message"],
            temperature=config["temperature"]
        )

        sessions[session_name] = response.session_id
        print(f"Created {session_name}: {response.session_id}")

    return sessions

# Usage
specialized_sessions = create_specialized_sessions()

# Use specialized sessions for specific tasks
coding_response = call_model(
    model="gpt-4",
    provider="openai",
    message="Write a binary search function",
    session_id=specialized_sessions["openai_coding"]
)

factual_response = call_model(
    model="gemini-2.0-flash",
    provider="google_genai",
    message="What is the population of Tokyo?",
    session_id=specialized_sessions["google_factual"]
)
```

## Cost Optimization Strategies

### Provider Cost Analysis

```python
def estimate_costs(message_length: int, response_length: int, model: str, provider: str):
    """Estimate costs for different providers (approximate rates)."""

    # Approximate costs per 1K tokens (as of 2024)
    cost_estimates = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        },
        "google_genai": {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003}
        },
        "anthropic": {
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
    }

    if provider in cost_estimates and model in cost_estimates[provider]:
        rates = cost_estimates[provider][model]

        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = message_length / 4
        output_tokens = response_length / 4

        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }

    return {"error": "Unknown provider/model combination"}

# Usage
cost = estimate_costs(
    message_length=100,
    response_length=500,
    model="gpt-4",
    provider="openai"
)
print(f"Estimated cost: ${cost['total_cost']:.6f}")
```

### Cost-Optimized Provider Selection

```python
def cost_optimized_call(message: str, quality_threshold: str = "medium"):
    """Select provider based on cost optimization and quality requirements."""

    quality_tiers = {
        "low": [
            ("gemini-2.0-flash", "google_genai"),
            ("claude-3-haiku", "anthropic"),
            ("gpt-3.5-turbo", "openai")
        ],
        "medium": [
            ("gemini-pro", "google_genai"),
            ("claude-3-sonnet", "anthropic"),
            ("gpt-3.5-turbo", "openai")
        ],
        "high": [
            ("claude-3-opus", "anthropic"),
            ("gpt-4", "openai"),
            ("claude-3-sonnet", "anthropic")
        ]
    }

    providers_to_try = quality_tiers.get(quality_threshold, quality_tiers["medium"])

    for model, provider in providers_to_try:
        try:
            response = call_model(
                model=model,
                provider=provider,
                message=message,
                temperature=0.3
            )

            # Estimate cost
            cost = estimate_costs(len(message), len(response.message), model, provider)

            return {
                "response": response,
                "model": model,
                "provider": provider,
                "estimated_cost": cost.get("total_cost", 0)
            }

        except Exception as e:
            print(f"Failed with {provider}:{model}: {e}")
            continue

    raise Exception("All providers failed")

# Usage
result = cost_optimized_call(
    "Explain machine learning in simple terms",
    quality_threshold="medium"
)

print(f"Used: {result['provider']}:{result['model']}")
print(f"Cost: ${result['estimated_cost']:.6f}")
print(f"Response: {result['response'].message}")
```
