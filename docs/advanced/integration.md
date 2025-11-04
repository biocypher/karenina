# System Integration

This guide explains how the Karenina library integrates with optional web-based components for enhanced workflow management.

## What is System Integration?

**Karenina** is a standalone Python library for LLM benchmarking. Everything documented in this guide works independently using pure Python code.

**Optional integration** with web-based components provides:
- **karenina-server**: REST API layer for remote access
- **karenina-gui**: Interactive web interface for visual workflows

The integration enables web-based benchmarking workflows while maintaining the library's standalone capabilities.

## Architecture Overview

Karenina uses a three-tier architecture:

```
┌─────────────────────────────────────────┐
│     karenina-gui (Frontend)             │
│     Interactive Web Interface           │
│  - Visual question extraction           │
│  - Template curation                    │
│  - Configuration management             │
│  - Real-time progress tracking          │
└────────────┬────────────────────────────┘
             │ REST API (/api/*)
┌────────────▼────────────────────────────┐
│   karenina-server (Middleware)          │
│   API Layer & Job Orchestration         │
│  - REST endpoints                       │
│  - Async job management                 │
│  - Background task processing           │
│  - WebSocket progress streaming         │
└────────────┬────────────────────────────┘
             │ Direct Python imports
┌────────────▼────────────────────────────┐
│      karenina (Core Library)            │
│      Pure Business Logic                │
│  - Question extraction                  │
│  - Template generation                  │
│  - Verification engine                  │
│  - Database persistence                 │
└─────────────────────────────────────────┘
```

**Key Points**:
- Each tier has clear responsibilities
- Core library works independently
- Server provides API access
- GUI provides visual interface

## Standalone Library Usage

All Karenina features work without server or GUI components:

```python
from pathlib import Path
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig

# Create benchmark
benchmark = Benchmark.create(
    name="Genomics Knowledge Benchmark",
    description="Testing LLM knowledge of genomics",
    version="1.0.0"
)

# Add questions
benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2",
    author={"name": "Pharma Curator"}
)

benchmark.add_question(
    question="Which chromosome contains the HBB gene?",
    raw_answer="Chromosome 11",
    author={"name": "Genetics Curator"}
)

# Configure verification
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True,
    deep_judgment_enabled=True
)

# Generate templates
benchmark.generate_all_templates(model_config=model_config)

# Run verification
results = benchmark.run_verification(config)

# Analyze results
success_count = sum(1 for r in results.values() if r.verify_result)
print(f"Success rate: {success_count}/{len(results)}")

# Save to database
db_path = Path("dbs/genomics.db")
benchmark.save(db_path)
```

**This works completely independently** - no server or GUI required.

## Server Integration (karenina-server)

The server adds API access for remote workflows.

### Features

**REST API**:
- Question extraction endpoints
- Template generation endpoints
- Verification endpoints
- Rubric management endpoints
- Database operations

**Job Management**:
- UUID-based job tracking
- Progress polling
- WebSocket streaming
- Cancellation support

**Static File Serving**:
- Serves built GUI in production
- Single-port deployment

### Installation

```bash
cd karenina-server
uv sync                    # Install dependencies
```

### Basic Usage

```bash
# Start development server
make serve

# Or use CLI directly
karenina-server serve --dev --port 8080
```

**API Documentation**: http://localhost:8080/docs

### API Example

```python
import requests

# Start verification job
response = requests.post("http://localhost:8080/api/start-verification", json={
    "templates": finished_templates,
    "config": config.model_dump()
})
job_id = response.json()["job_id"]

# Poll for progress
while True:
    progress = requests.get(
        f"http://localhost:8080/api/verification-progress/{job_id}"
    ).json()

    if progress["status"] == "completed":
        results = progress["results"]
        break
    elif progress["status"] == "failed":
        print(f"Error: {progress['error']}")
        break

    time.sleep(1)
```

### Production Deployment

```bash
# Build GUI first
cd karenina-gui
npm run build

# Start production server (serves GUI + API)
cd karenina-server
karenina-server serve --host 0.0.0.0 --port 8080
```

**Access**:
- GUI: http://localhost:8080
- API: http://localhost:8080/api/*

## GUI Integration (karenina-gui)

The GUI provides an interactive web interface.

### Features

**Template Generation Tab**:
- File upload (Excel/CSV/TSV)
- Question extraction
- Template generation
- Real-time progress tracking

**Curator Tab**:
- Code editor with syntax highlighting
- Metadata editing
- Few-shot example management
- Template status tracking

**Benchmark Tab**:
- Configuration management
- Verification execution
- Results visualization
- Export options

**Docs Tab**:
- Embedded documentation

### Installation

```bash
cd karenina-gui
npm install                # Install dependencies
```

### Development Mode

```bash
# Start GUI dev server (requires server running)
npm run dev

# In separate terminal, start backend
cd karenina-server
make serve
```

**Access**: http://localhost:5173

**Note**: Development mode proxies API calls to server at http://localhost:8080

### Production Mode

```bash
# Build GUI
npm run build

# GUI served by karenina-server in production mode
cd karenina-server
karenina-server serve --host 0.0.0.0 --port 8080
```

## Integration Benefits

### Standalone Library Benefits

**Advantages**:
- No dependencies beyond Python packages
- Scriptable workflows
- CI/CD integration
- Jupyter notebook support
- Full programmatic control

**Use Cases**:
- Automated testing pipelines
- Research notebooks
- Batch processing
- Scripted evaluations

### Server Integration Benefits

**Advantages**:
- Remote access via REST API
- Multi-user support (future)
- Language-agnostic clients
- Job queue management
- Background processing

**Use Cases**:
- Team collaboration
- Remote benchmarking
- Service integration
- API consumption

### GUI Integration Benefits

**Advantages**:
- Visual workflow management
- Real-time progress tracking
- Interactive data exploration
- No coding required
- Template curation tools

**Use Cases**:
- Non-technical users
- Interactive exploration
- Template refinement
- Results visualization

## Configuration

### Environment Variables

All tiers share environment variables:

```bash
# LLM API Keys
OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="AIza..."
ANTHROPIC_API_KEY="sk-ant-..."
OPENROUTER_API_KEY="sk-or-..."

# Database
DB_PATH="dbs/karenina.db"

# Features
EMBEDDING_CHECK="true"
EMBEDDING_CHECK_MODEL="all-MiniLM-L6-v2"
EMBEDDING_CHECK_THRESHOLD="0.85"
ABSTENTION_CHECK_ENABLED="true"

# Server (optional)
KARENINA_WEBAPP_DIR="/path/to/karenina-gui/dist"
```

### Model Configuration

Default model settings apply across all tiers:

```python
# Standalone library
model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

# Server API (same structure)
requests.post("/api/start-verification", json={
    "config": {
        "answering_models": [
            {
                "model_name": "gpt-4.1-mini",
                "model_provider": "openai",
                "temperature": 0.0
            }
        ]
    }
})

# GUI (visual configuration)
# Select model: gpt-4.1-mini (OpenAI)
# Temperature: 0.0
```

## Data Persistence

### Database Storage

All tiers use the same SQLite database:

```python
from pathlib import Path
from karenina import Benchmark

# Save from library
db_path = Path("dbs/genomics.db")
benchmark.save(db_path)

# Load from library
loaded = Benchmark.load(db_path, "Genomics Knowledge Benchmark")

# Server API uses same database
# GET /api/database/list?db_path=dbs/genomics.db
# POST /api/database/load

# GUI connects to same database via server
```

**Key Points**:
- Single source of truth
- Shared database across tiers
- SQLite for simplicity
- File-based storage

### Checkpoint Files

Checkpoint files (JSON-LD format) work across tiers:

```python
from pathlib import Path

# Save checkpoint from library
checkpoint_path = Path("checkpoints/genomics_checkpoint.json")
benchmark.save_checkpoint(checkpoint_path)

# Load in GUI via file upload
# Template Generation Tab -> Load Checkpoint

# Export from GUI
# Curator Tab -> File Manager -> Save Checkpoint
```

## Choosing an Integration Level

### Use Standalone Library When:

- Automating benchmarking workflows
- Integrating with existing pipelines
- Running in CI/CD environments
- Working in Jupyter notebooks
- Scripting batch evaluations

### Add Server When:

- Need remote API access
- Building custom clients
- Supporting multiple users
- Integrating with other services
- Require background job processing

### Add GUI When:

- Need visual workflow management
- Supporting non-technical users
- Require interactive exploration
- Curating templates manually
- Visualizing results interactively

## Complete Example: Three-Tier Workflow

This example shows the same workflow across all tiers.

### Step 1: Create Benchmark (Library)

```python
from karenina import Benchmark

benchmark = Benchmark.create(
    name="Drug Target Benchmark",
    description="Testing drug target knowledge",
    version="1.0.0"
)

benchmark.add_question(
    question="What is the approved drug target of Venetoclax?",
    raw_answer="BCL2"
)

# Save to database
from pathlib import Path
benchmark.save(Path("dbs/drugs.db"))
```

### Step 2: Generate Templates (Server)

```python
import requests

# Load benchmark via API
response = requests.post("http://localhost:8080/api/database/load", json={
    "db_path": "dbs/drugs.db",
    "benchmark_name": "Drug Target Benchmark"
})

# Start template generation
response = requests.post("http://localhost:8080/api/generate-answer-templates", json={
    "questions": benchmark.export_questions(),
    "config": {
        "model_name": "gpt-4.1-mini",
        "model_provider": "openai",
        "temperature": 0.0
    }
})

job_id = response.json()["job_id"]

# Poll for completion
while True:
    progress = requests.get(
        f"http://localhost:8080/api/generation-progress/{job_id}"
    ).json()

    if progress["status"] == "completed":
        templates = progress["results"]
        break

    time.sleep(1)
```

### Step 3: Curate Templates (GUI)

1. Open http://localhost:8080
2. Navigate to **Curator** tab
3. Load checkpoint or connect to database
4. Review generated templates
5. Edit Pydantic classes as needed
6. Mark templates as finished
7. Save checkpoint

### Step 4: Run Verification (Library)

```python
# Load curated checkpoint
checkpoint = benchmark.load_checkpoint(
    Path("checkpoints/drugs_curated.json")
)

# Run verification
from karenina.schemas import ModelConfig, VerificationConfig

model_config = ModelConfig(
    model_name="gpt-4.1-mini",
    model_provider="openai",
    temperature=0.0
)

config = VerificationConfig(
    answering_models=[model_config],
    parsing_models=[model_config],
    replicate_count=3,
    rubric_enabled=True
)

results = benchmark.run_verification(config)

# Save results
benchmark.save_results(Path("dbs/drugs.db"), results)
```

### Step 5: Analyze Results (GUI)

1. Navigate to **Benchmark** tab
2. Load results from database
3. View results table
4. Filter by status
5. Export to CSV/JSON

## Best Practices

### Standalone Development

**Do**:
- Use library for scripted workflows
- Save checkpoints frequently
- Use database for persistence
- Test in notebooks first

**Don't**:
- Mix library and API calls unnecessarily
- Duplicate data across formats
- Skip database backups

### Server Development

**Do**:
- Use job tracking for long operations
- Handle errors gracefully
- Monitor background jobs
- Use WebSocket for real-time updates

**Don't**:
- Poll too frequently (respect rate limits)
- Ignore job cleanup
- Mix sync and async patterns

### GUI Development

**Do**:
- Use GUI for interactive workflows
- Leverage real-time progress tracking
- Save checkpoints before major changes
- Export results regularly

**Don't**:
- Rely on session state (cleared on refresh)
- Edit templates without backup
- Skip configuration validation

### Hybrid Workflows

**Do**:
- Generate templates via library/server
- Curate in GUI
- Run verification via library
- Analyze in GUI

**Don't**:
- Switch tiers mid-workflow unnecessarily
- Lose track of data location
- Skip synchronization points

## Troubleshooting

### Server Connection Issues

**Problem**: GUI can't connect to server

**Solutions**:
- Verify server is running: `curl http://localhost:8080/api/health`
- Check CORS configuration
- Verify proxy settings in `vite.config.ts`
- Check firewall rules

### Database Conflicts

**Problem**: Database locked or inaccessible

**Solutions**:
- Close other connections
- Check file permissions
- Use absolute paths
- Verify database exists

### Job Tracking Issues

**Problem**: Jobs stuck in "running" state

**Solutions**:
- Check server logs for errors
- Verify background workers running
- Cancel and restart job
- Check resource availability

### WebSocket Issues

**Problem**: Real-time progress not updating

**Solutions**:
- Verify WebSocket endpoint accessible
- Check browser console for errors
- Fall back to polling mode
- Verify server WebSocket support

## Related Documentation

- **Core Library**: All feature guides in `docs/` directory
- **Server API**: Auto-generated docs at http://localhost:8080/docs
- **GUI**: Embedded documentation in Docs tab
- **Architecture**: See `.agents/architecture.md` (internal docs)

## Summary

Karenina provides flexible integration options:

1. **Standalone Library**: Full-featured Python library for scripted workflows
2. **Server Integration**: REST API for remote access and job management
3. **GUI Integration**: Interactive web interface for visual workflows

**Choose based on your needs**:
- Research and automation → Standalone library
- Team collaboration → Add server
- Non-technical users → Add GUI

All tiers share the same data models, database, and configuration - ensuring consistency across workflows.
