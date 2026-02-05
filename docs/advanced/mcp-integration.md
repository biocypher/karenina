# MCP Integration

This guide explains how to integrate Model Context Protocol (MCP) servers to provide tool access for LLMs during verification.

## What is MCP?

**Model Context Protocol (MCP)** is a standardized protocol that enables LLMs to access external tools and data sources. MCP servers provide tools that LLMs can invoke during answer generation, such as:

- Web search
- Database queries
- File system operations
- API calls
- Code execution
- Custom domain-specific tools

**Key Benefits**:

- Extend LLM capabilities beyond text generation
- Access real-time data during verification
- Standardized tool invocation protocol
- Modular tool integration

## Why Use MCP with Karenina?

MCP integration allows LLMs to access external information when answering benchmark questions:

**Use Cases**:

- **Current information**: Search web for recent drug approvals
- **Database access**: Query genomics databases for gene information
- **File operations**: Read configuration files or data files
- **API integration**: Call external APIs for real-time data
- **Custom tools**: Domain-specific tools for specialized benchmarks

**Example**: A benchmark question asks "What is the current FDA approval status of drug X?" The LLM can use an MCP web search tool to find the latest information instead of relying on training data.

## MCP Server Structure

An MCP server provides:

1. **Health Check**: Endpoint to verify server is running
2. **Tool Discovery**: List available tools and their schemas
3. **Tool Invocation**: Execute tools with parameters

```
MCP Server (http://localhost:3000/mcp)
├── GET  /health          # Server status
├── GET  /tools           # Available tools
└── POST /invoke          # Execute a tool
```

## Configuration

### Basic Setup

Configure MCP integration via `VerificationConfig`:

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Benchmark",
    description="Testing genomics knowledge with tool access",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the latest research on BCL2 protein function?",
    raw_answer="BCL2 regulates apoptosis",
    author={"name": "Research Curator"}
)

# Generate templates
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config)

# Configure verification with MCP server
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    mcp_server_url="http://localhost:3000/mcp"  # MCP server URL
)

# Run verification (LLM can now use MCP tools)
results = benchmark.run_verification(config)
```

### Multiple MCP Servers

Configure multiple MCP servers for different tool categories:

```python
from karenina.schemas import VerificationConfig, ModelConfig

model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# Multiple MCP servers
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    mcp_servers=[
        {
            "url": "http://localhost:3000/mcp",
            "name": "Search Tools"
        },
        {
            "url": "http://localhost:3001/mcp",
            "name": "Database Tools"
        }
    ]
)

# LLM has access to tools from both servers
results = benchmark.run_verification(config)
```

## Example MCP Tools

### Web Search Tool

Enables LLMs to search for current information:

```json
{
  "name": "web_search",
  "description": "Search the web for information",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      }
    },
    "required": ["query"]
  }
}
```

**Example usage**: LLM searches for "latest BCL2 protein research" to answer a genomics question with recent findings.

### Database Query Tool

Allows LLMs to query databases:

```json
{
  "name": "query_genomics_db",
  "description": "Query genomics database for gene information",
  "parameters": {
    "type": "object",
    "properties": {
      "gene_name": {
        "type": "string",
        "description": "Gene symbol (e.g., HBB, BCL2)"
      }
    },
    "required": ["gene_name"]
  }
}
```

**Example usage**: LLM queries database for "BCL2" to get official gene information, protein function, and chromosome location.

### File Read Tool

Enables LLMs to read data files:

```json
{
  "name": "read_data_file",
  "description": "Read contents of a data file",
  "parameters": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to data file"
      }
    },
    "required": ["file_path"]
  }
}
```

**Example usage**: LLM reads a drug-target database file to answer questions about approved therapeutics.

## Complete Example

This example shows MCP integration for a genomics benchmark with web search:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Step 1: Create benchmark with questions requiring current data
benchmark = Benchmark.create(
    name="Current Genomics Research",
    description="Testing knowledge of recent genomics discoveries",
    version="1.0.0"
)

# Questions that benefit from tool access
benchmark.add_question(
    question="What are the latest findings on BCL2's role in cancer therapy?",
    raw_answer="BCL2 inhibition shows promise in treating certain cancers",
    author={"name": "Oncology Researcher"}
)

benchmark.add_question(
    question="What is the current status of CRISPR therapies for hemoglobin disorders?",
    raw_answer="CRISPR treatments for sickle cell disease are in clinical trials",
    author={"name": "Gene Therapy Researcher"}
)

# Step 2: Generate templates (one-time)
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

benchmark.generate_all_templates(model_config=model_config)

# Step 3: Configure verification with MCP web search tool
# (Assumes MCP server running at localhost:3000 with search tool)
config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    mcp_server_url="http://localhost:3000/mcp"  # Web search MCP server
)

# Step 4: Run verification with tool access
# LLM can search web for current information
results = benchmark.run_verification(config)

# Step 5: Analyze results
for result in results.results:
    question = benchmark.questions[question_id]
    print(f"\nQuestion: {question.question}")
    print(f"Verification: {'✓ PASS' if result.verify_result else '✗ FAIL'}")

    # Check if tools were used
    if hasattr(result, 'tools_used') and result.tools_used:
        print(f"Tools invoked: {result.tools_used}")

# Save results
benchmark.save_to_db(Path("dbs/genomics_with_mcp.db"))
```

## Validating MCP Servers

Before running verification, validate MCP server connectivity:

```python
from karenina.llm.mcp_utils import validate_mcp_server

# Test MCP server
server_url = "http://localhost:3000/mcp"
is_valid, capabilities = validate_mcp_server(server_url)

if is_valid:
    print("✓ MCP server is reachable")
    print(f"Available tools:")
    for tool in capabilities.get('tools', []):
        print(f"  - {tool['name']}: {tool['description']}")
else:
    print("✗ MCP server validation failed")
    print("Ensure server is running and accessible")
```

## Discovering Available Tools

Query an MCP server to see what tools it provides:

```python
from karenina.llm.mcp_utils import get_mcp_tools

# Discover tools
server_url = "http://localhost:3000/mcp"
tools = get_mcp_tools(server_url)

print(f"Discovered {len(tools)} tools:")
for tool in tools:
    print(f"\nTool: {tool['name']}")
    print(f"Description: {tool['description']}")
    print(f"Parameters: {tool['parameters']}")
```

## Use Cases

### Use Case 1: Current Information Access

**Scenario**: Benchmark tests LLM knowledge of recent drug approvals.

**Setup**:

- Deploy MCP server with web search tool
- Configure verification with MCP server URL
- Questions ask about recent FDA approvals

**Benefit**: LLM can search for current information instead of relying on training data cutoff.

### Use Case 2: Database Integration

**Scenario**: Questions require querying a genomics database.

**Setup**:

- Deploy MCP server with database query tool
- Configure database connection in MCP server
- Questions ask about specific genes

**Benefit**: LLM gets accurate, up-to-date gene information from authoritative database.

### Use Case 3: File-Based Data

**Scenario**: Benchmark uses data files with drug-target mappings.

**Setup**:

- Deploy MCP server with file read tool
- Store drug-target data in structured files
- Configure file system access permissions

**Benefit**: LLM reads data files to answer questions accurately without relying on memorized facts.

### Use Case 4: API Integration

**Scenario**: Questions require real-time API data.

**Setup**:

- Deploy MCP server with API call tools
- Configure API keys and endpoints
- Questions ask about live data

**Benefit**: LLM calls APIs to fetch current data during verification.

## Creating a Simple MCP Server

Example MCP server with a genomics database query tool:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Mock genomics database
GENOMICS_DB = {
    "BCL2": {
        "full_name": "B-cell lymphoma 2",
        "chromosome": "18",
        "function": "Regulates apoptosis"
    },
    "HBB": {
        "full_name": "Hemoglobin subunit beta",
        "chromosome": "11",
        "function": "Oxygen transport"
    }
}

class ToolInvocation(BaseModel):
    tool_name: str
    parameters: dict

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/tools")
def list_tools():
    return {
        "tools": [
            {
                "name": "query_gene",
                "description": "Query genomics database for gene information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_symbol": {
                            "type": "string",
                            "description": "Gene symbol (e.g., BCL2, HBB)"
                        }
                    },
                    "required": ["gene_symbol"]
                }
            }
        ]
    }

@app.post("/invoke")
def invoke_tool(invocation: ToolInvocation):
    if invocation.tool_name == "query_gene":
        gene = invocation.parameters.get("gene_symbol", "").upper()
        if gene in GENOMICS_DB:
            return {"result": GENOMICS_DB[gene]}
        return {"error": f"Gene {gene} not found in database"}

    return {"error": "Unknown tool"}

# Run: uvicorn mcp_server:app --port 3000
```

**Usage**:
```bash
# Start MCP server
uvicorn mcp_server:app --port 3000

# In another terminal, run Karenina verification with MCP
python verify_with_mcp.py
```

## Anthropic Prompt Caching

When using Anthropic models (Claude) with MCP tools via the `langchain` interface, **prompt caching is enabled by default** to reduce costs and latency. This caches repetitive prompt content like system prompts, tool definitions, and conversation history on Anthropic's servers.

### How It Works

1. **First request**: System prompt, tools, and the user message are sent to the API and cached
2. **Subsequent requests**: Cached content is retrieved rather than reprocessed
3. **Cache expiration**: Content expires after the TTL (5 minutes or 1 hour)

### Configuration

Prompt caching is configured via `AgentMiddlewareConfig` in `ModelConfig`:

```python
from karenina.schemas import ModelConfig
from karenina.schemas.config import AgentMiddlewareConfig, PromptCachingConfig

model_config = ModelConfig(
    id="agent-claude",
    model_provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    temperature=0.0,
    interface="langchain",
    mcp_urls_dict={"biocontext": "https://mcp.biocontext.ai/mcp/"},
    agent_middleware=AgentMiddlewareConfig(
        prompt_caching=PromptCachingConfig(
            enabled=True,           # Default: True for Anthropic models
            ttl="5m",               # Cache lifetime: "5m" (5 minutes) or "1h" (1 hour)
            min_messages_to_cache=0,  # Min messages before caching starts
            unsupported_model_behavior="warn",  # "ignore", "warn", or "raise"
        )
    )
)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable prompt caching |
| `ttl` | `"5m"` | Cache time-to-live: `"5m"` or `"1h"` |
| `min_messages_to_cache` | `0` | Minimum messages before caching activates |
| `unsupported_model_behavior` | `"warn"` | Behavior for non-Anthropic models |

### Disabling Prompt Caching

To disable prompt caching for Anthropic models:

```python
agent_middleware=AgentMiddlewareConfig(
    prompt_caching=PromptCachingConfig(enabled=False)
)
```

### Requirements

- **Provider**: `anthropic` only
- **Interface**: `langchain` only
- **Dependency**: `langchain-anthropic` must be installed

Prompt caching does **not** provide conversation memory - it only reduces API costs by caching tokens. For conversation persistence, use a checkpointer.

See the [LangChain documentation](https://docs.langchain.com/oss/python/integrations/middleware/anthropic#prompt-caching) for more details.

## Best Practices

### Server Configuration

**Do**:

- Validate MCP server before verification
- Use HTTPS in production
- Implement authentication for MCP servers
- Set appropriate timeout limits
- Log tool invocations for debugging

**Don't**:

- Expose MCP servers publicly without authentication
- Allow unrestricted file system access
- Skip server validation before use
- Use untrusted MCP servers

### Tool Design

**Do**:

- Provide clear tool descriptions
- Use typed parameters with JSON schema
- Return structured data
- Handle errors gracefully
- Document tool capabilities

**Don't**:

- Create tools with side effects (prefer read-only)
- Skip parameter validation
- Return unstructured text
- Allow dangerous operations without safeguards

### Security

**Do**:

- Validate all tool parameters
- Restrict tool permissions (principle of least privilege)
- Implement rate limiting
- Monitor tool usage
- Use network firewalls

**Don't**:

- Trust tool input without validation
- Grant excessive permissions
- Skip logging
- Ignore security warnings

## Troubleshooting

### Issue: MCP Server Not Reachable

**Error**: `Connection refused` or timeout errors

**Cause**: MCP server not running or wrong URL.

**Solutions**:
```bash
# Check server is running
curl http://localhost:3000/mcp/health

# Verify port and URL
ps aux | grep mcp

# Check firewall rules
```

### Issue: Tool Not Available

**Error**: `Tool 'web_search' not found`

**Cause**: Tool not registered in MCP server.

**Solution**:
```bash
# List available tools
curl http://localhost:3000/mcp/tools

# Verify tool name spelling matches exactly
```

### Issue: Tool Invocation Fails

**Error**: `Tool invocation failed: invalid parameters`

**Cause**: Parameters don't match tool schema.

**Solution**:
```python
# Check tool schema
from karenina.llm.mcp_utils import get_mcp_tools

tools = get_mcp_tools("http://localhost:3000/mcp")
for tool in tools:
    if tool['name'] == 'web_search':
        print("Required parameters:", tool['parameters'])
```

### Issue: Verification Slower with MCP

**Symptom**: Verification takes much longer with MCP enabled.

**Cause**: Tool invocations add latency.

**Solution**:

- Use faster MCP servers (local is better than remote)
- Cache tool results when possible
- Reduce network latency
- Set appropriate timeouts

## Limitations

**Current Limitations**:

- MCP integration primarily designed for server/GUI deployment
- Standalone library support is experimental
- Tool invocation tracking may be limited
- Some providers may not support function calling

**Best Use**:

- Use with karenina-server and karenina-gui for full features
- Standalone library works but with reduced visibility into tool usage
- Consider manual traces for reproducible testing instead

## Related Documentation

- **System Integration**: Full integration with server and GUI
- **Configuration**: Model and verification configuration
- **Manual Traces**: Alternative for reproducible testing
- **API Reference**: Complete API documentation

## Summary

MCP integration enables:

1. **Tool access** - LLMs can use external tools during verification
2. **Current data** - Access real-time information beyond training data
3. **Database queries** - Query structured databases
4. **File operations** - Read data from files
5. **API calls** - Integrate with external APIs

**Configure** by setting `mcp_server_url` in `VerificationConfig` and ensure MCP server is running and accessible.

**Note**: MCP integration is most powerful when used with the full karenina-server and karenina-gui stack. Standalone library support exists but is more limited.
