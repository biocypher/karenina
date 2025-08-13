# Quickstart

Minimal end-to-end using only `Benchmark`.

## Install

```bash
pip install karenina
```

## Create, add questions, save

```python
from karenina.benchmark import Benchmark

benchmark = Benchmark.create(
    name="Quickstart",
    description="Minimal demo"
)

qid = benchmark.add_question(
    question="What is the capital of France?",
    raw_answer="Paris"
)

benchmark.save("quickstart.jsonld")
print(benchmark)
```

Expected output (similar):

```text
Benchmark 'Quickstart' (1 questions, 0.0% complete)
```

## Load an existing benchmark and inspect

```python
from karenina.benchmark import Benchmark

bench = Benchmark.load("quickstart.jsonld")
print(len(bench))
print(bench.get_progress())
print(bench.get_question_ids())
```

## Next steps

- See the Benchmark Guide chapters for templates, verification, and results.
- Full API: API Reference.
