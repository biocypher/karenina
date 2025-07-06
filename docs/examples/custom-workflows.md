# Custom Workflows

Advanced examples demonstrating custom workflows and specialized use cases with Karenina.

## Advanced Question Processing

### Domain-Specific Question Filtering

```python
import re
from karenina.questions.extractor import extract_questions_from_file
from karenina.schemas.question_class import Question

def filter_questions_by_domain(questions: list, domain: str) -> list:
    """Filter questions by domain-specific keywords and patterns."""

    domain_patterns = {
        'mathematics': {
            'keywords': ['calculate', 'compute', 'solve', 'equation', 'formula', 'number'],
            'patterns': [r'\d+[\+\-\*/]\d+', r'x\s*=', r'f\(x\)', r'\b\d+%']
        },
        'science': {
            'keywords': ['theory', 'experiment', 'hypothesis', 'molecule', 'energy', 'force'],
            'patterns': [r'chemical formula', r'physics', r'biology', r'chemistry']
        },
        'geography': {
            'keywords': ['capital', 'country', 'city', 'continent', 'ocean', 'mountain'],
            'patterns': [r'located in', r'borders', r'population of']
        },
        'programming': {
            'keywords': ['function', 'variable', 'loop', 'algorithm', 'code', 'syntax'],
            'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+', r'for\s+\w+\s+in']
        }
    }

    if domain not in domain_patterns:
        return questions

    criteria = domain_patterns[domain]
    filtered = []

    for question in questions:
        text_lower = question.question.lower()

        # Check keywords
        keyword_match = any(keyword in text_lower for keyword in criteria['keywords'])

        # Check patterns
        pattern_match = any(re.search(pattern, question.question, re.IGNORECASE)
                          for pattern in criteria['patterns'])

        if keyword_match or pattern_match:
            # Add domain tag
            if domain not in question.tags:
                question.tags.append(domain)
            filtered.append(question)

    return filtered

def create_domain_specific_dataset(input_file: str, domain: str, output_file: str):
    """Create domain-specific question dataset."""

    print(f"Creating {domain} dataset from {input_file}")

    # Extract all questions
    all_questions = extract_questions_from_file(input_file, "Question", "Answer")
    print(f"Extracted {len(all_questions)} total questions")

    # Filter by domain
    domain_questions = filter_questions_by_domain(all_questions, domain)
    print(f"Filtered to {len(domain_questions)} {domain} questions")

    # Generate domain-specific file
    from karenina.questions.extractor import generate_questions_file
    generate_questions_file(domain_questions, output_file)

    print(f"Generated {output_file}")
    return domain_questions

# Usage examples
math_questions = create_domain_specific_dataset(
    "data/mixed_questions.xlsx",
    "mathematics",
    "math_questions.py"
)

science_questions = create_domain_specific_dataset(
    "data/mixed_questions.xlsx",
    "science",
    "science_questions.py"
)
```

### Question Complexity Analysis

```python
import statistics
from textstat import flesch_reading_ease, syllable_count

def analyze_question_complexity(questions: list) -> dict:
    """Analyze complexity metrics for questions."""

    metrics = {
        'length_stats': {},
        'readability_stats': {},
        'complexity_distribution': {},
        'question_types': {}
    }

    lengths = []
    readability_scores = []
    word_counts = []
    question_types = {'what': 0, 'how': 0, 'why': 0, 'when': 0, 'where': 0, 'other': 0}

    for question in questions:
        text = question.question

        # Length metrics
        lengths.append(len(text))
        word_count = len(text.split())
        word_counts.append(word_count)

        # Readability
        try:
            readability = flesch_reading_ease(text)
            readability_scores.append(readability)
        except:
            pass

        # Question type
        text_lower = text.lower()
        if text_lower.startswith('what'):
            question_types['what'] += 1
        elif text_lower.startswith('how'):
            question_types['how'] += 1
        elif text_lower.startswith('why'):
            question_types['why'] += 1
        elif text_lower.startswith('when'):
            question_types['when'] += 1
        elif text_lower.startswith('where'):
            question_types['where'] += 1
        else:
            question_types['other'] += 1

    # Calculate statistics
    if lengths:
        metrics['length_stats'] = {
            'mean': statistics.mean(lengths),
            'median': statistics.median(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'std': statistics.stdev(lengths) if len(lengths) > 1 else 0
        }

    if word_counts:
        metrics['word_count_stats'] = {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'min': min(word_counts),
            'max': max(word_counts)
        }

    if readability_scores:
        metrics['readability_stats'] = {
            'mean': statistics.mean(readability_scores),
            'median': statistics.median(readability_scores)
        }

    metrics['question_types'] = question_types

    return metrics

def select_questions_by_complexity(questions: list, complexity_level: str) -> list:
    """Select questions based on complexity level."""

    analysis = analyze_question_complexity(questions)

    if not analysis['word_count_stats']:
        return questions

    mean_words = analysis['word_count_stats']['mean']
    std_words = analysis['word_count_stats']['std'] if 'std' in analysis['word_count_stats'] else 0

    complexity_thresholds = {
        'simple': mean_words - std_words,
        'medium': mean_words,
        'complex': mean_words + std_words
    }

    threshold = complexity_thresholds.get(complexity_level, mean_words)

    selected = []
    for question in questions:
        word_count = len(question.question.split())

        if complexity_level == 'simple' and word_count <= threshold:
            selected.append(question)
        elif complexity_level == 'medium' and abs(word_count - threshold) <= std_words:
            selected.append(question)
        elif complexity_level == 'complex' and word_count >= threshold:
            selected.append(question)

    return selected

# Usage
questions = extract_questions_from_file("data/questions.xlsx", "Question", "Answer")
complexity_analysis = analyze_question_complexity(questions)

print("Question Analysis:")
print(f"Average length: {complexity_analysis['length_stats']['mean']:.1f} characters")
print(f"Average words: {complexity_analysis['word_count_stats']['mean']:.1f}")
print("Question types:", complexity_analysis['question_types'])

# Select by complexity
simple_questions = select_questions_by_complexity(questions, 'simple')
complex_questions = select_questions_by_complexity(questions, 'complex')

print(f"Simple questions: {len(simple_questions)}")
print(f"Complex questions: {len(complex_questions)}")
```

## Advanced Template Generation

### Multi-Model Template Ensemble

```python
from karenina.answers.generator import generate_answer_template
from karenina.utils.code_parser import extract_and_combine_codeblocks
import json

def generate_ensemble_templates(question: str, question_json: str) -> dict:
    """Generate templates using multiple models and combine insights."""

    models_config = [
        {"model": "gpt-4", "provider": "openai", "weight": 0.4},
        {"model": "claude-3-sonnet", "provider": "anthropic", "weight": 0.3},
        {"model": "gemini-pro", "provider": "google_genai", "weight": 0.3}
    ]

    templates = {}

    for config in models_config:
        try:
            print(f"Generating template with {config['model']}...")

            template_code = generate_answer_template(
                question=question,
                question_json=question_json,
                model=config["model"],
                model_provider=config["provider"],
                temperature=0.1  # Low temperature for consistent structure
            )

            code_blocks = extract_and_combine_codeblocks(template_code)

            # Analyze generated template
            field_analysis = analyze_template_structure(code_blocks)

            templates[config["model"]] = {
                "code": code_blocks,
                "analysis": field_analysis,
                "weight": config["weight"]
            }

        except Exception as e:
            print(f"Error with {config['model']}: {e}")

    # Combine insights from multiple templates
    combined_template = combine_template_insights(templates)

    return templates, combined_template

def analyze_template_structure(code: str) -> dict:
    """Analyze the structure of a generated template."""

    import ast
    import re

    analysis = {
        'fields': [],
        'field_types': {},
        'constraints': [],
        'imports': []
    }

    try:
        # Parse the code
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Answer":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        field_name = item.target.id
                        analysis['fields'].append(field_name)

                        # Extract type annotation
                        if item.annotation:
                            analysis['field_types'][field_name] = ast.unparse(item.annotation)

        # Extract Field constraints
        field_patterns = re.findall(r'Field\([^)]+\)', code)
        analysis['constraints'] = field_patterns

        # Extract imports
        import_patterns = re.findall(r'from\s+\S+\s+import\s+\S+|import\s+\S+', code)
        analysis['imports'] = import_patterns

    except Exception as e:
        analysis['error'] = str(e)

    return analysis

def combine_template_insights(templates: dict) -> str:
    """Combine insights from multiple template generations."""

    # Collect all fields from all templates
    all_fields = set()
    field_type_votes = {}

    for model, template_info in templates.items():
        analysis = template_info['analysis']
        weight = template_info['weight']

        for field in analysis.get('fields', []):
            all_fields.add(field)

            field_type = analysis['field_types'].get(field, 'str')
            if field not in field_type_votes:
                field_type_votes[field] = {}

            if field_type not in field_type_votes[field]:
                field_type_votes[field][field_type] = 0

            field_type_votes[field][field_type] += weight

    # Create combined template
    combined_code = '''from pydantic import BaseModel, Field
from karenina.schemas.answer_class import BaseAnswer
from typing import Optional

class Answer(BaseAnswer):
'''

    for field in sorted(all_fields):
        # Choose most voted type
        if field in field_type_votes:
            best_type = max(field_type_votes[field].items(), key=lambda x: x[1])[0]
        else:
            best_type = "str"

        combined_code += f'    {field}: {best_type} = Field(description="{field} field")\n'

    return combined_code

# Usage
question = "What are the key factors affecting climate change?"
question_obj = Question(
    id="climate_q1",
    question=question,
    raw_answer="Multiple factors including greenhouse gases, deforestation, etc.",
    tags=["environment"]
)

templates, combined = generate_ensemble_templates(question, question_obj.model_dump_json())

print("Generated templates from multiple models:")
for model, info in templates.items():
    print(f"\n{model}:")
    print(f"  Fields: {info['analysis'].get('fields', [])}")

print("\nCombined template:")
print(combined)
```

### Adaptive Template Generation

```python
def adaptive_template_generation(questions: list, feedback_data: dict = None) -> dict:
    """Generate templates that adapt based on question characteristics and feedback."""

    templates = {}
    generation_strategies = {}

    for question in questions:
        # Analyze question characteristics
        question_analysis = analyze_single_question(question)

        # Select generation strategy
        strategy = select_generation_strategy(question_analysis, feedback_data)
        generation_strategies[question.id] = strategy

        # Generate template with selected strategy
        template_code = generate_answer_template(
            question=question.question,
            question_json=question.model_dump_json(),
            model=strategy['model'],
            model_provider=strategy['provider'],
            temperature=strategy['temperature'],
            custom_system_prompt=strategy.get('system_prompt')
        )

        # Execute and store template
        code_blocks = extract_and_combine_codeblocks(template_code)
        local_ns = {}
        exec(code_blocks, globals(), local_ns)
        templates[question.id] = local_ns["Answer"]

    return templates, generation_strategies

def analyze_single_question(question: Question) -> dict:
    """Analyze characteristics of a single question."""

    text = question.question.lower()

    analysis = {
        'type': 'other',
        'complexity': 'medium',
        'domain': 'general',
        'requires_computation': False,
        'requires_explanation': False,
        'word_count': len(question.question.split())
    }

    # Question type
    if text.startswith('what'):
        analysis['type'] = 'factual'
    elif text.startswith('how'):
        analysis['type'] = 'procedural'
    elif text.startswith('why'):
        analysis['type'] = 'explanatory'
        analysis['requires_explanation'] = True
    elif text.startswith(('calculate', 'compute', 'solve')):
        analysis['type'] = 'computational'
        analysis['requires_computation'] = True

    # Domain detection
    if any(word in text for word in ['math', 'calculate', 'number', 'equation']):
        analysis['domain'] = 'mathematics'
    elif any(word in text for word in ['science', 'theory', 'experiment']):
        analysis['domain'] = 'science'
    elif any(word in text for word in ['code', 'program', 'function', 'algorithm']):
        analysis['domain'] = 'programming'

    # Complexity
    if analysis['word_count'] < 10:
        analysis['complexity'] = 'simple'
    elif analysis['word_count'] > 20:
        analysis['complexity'] = 'complex'

    return analysis

def select_generation_strategy(question_analysis: dict, feedback_data: dict = None) -> dict:
    """Select optimal generation strategy based on question analysis."""

    # Default strategy
    strategy = {
        'model': 'gemini-2.0-flash',
        'provider': 'google_genai',
        'temperature': 0.2
    }

    # Adjust based on question type
    if question_analysis['type'] == 'computational':
        strategy.update({
            'model': 'gpt-4',
            'provider': 'openai',
            'temperature': 0.0,
            'system_prompt': '''Generate a detailed answer template for computational questions.
            Include fields for:
            - The numerical result
            - Step-by-step calculation
            - Method used
            - Confidence in the answer
            - Units if applicable'''
        })

    elif question_analysis['type'] == 'explanatory':
        strategy.update({
            'model': 'claude-3-sonnet',
            'provider': 'anthropic',
            'temperature': 0.3,
            'system_prompt': '''Generate a comprehensive answer template for explanatory questions.
            Include fields for:
            - Main explanation
            - Supporting evidence
            - Examples
            - Complexity level
            - Certainty of explanation'''
        })

    elif question_analysis['domain'] == 'programming':
        strategy.update({
            'model': 'gpt-4',
            'provider': 'openai',
            'temperature': 0.1,
            'system_prompt': '''Generate an answer template for programming questions.
            Include fields for:
            - Code solution
            - Explanation of approach
            - Time complexity
            - Space complexity
            - Alternative approaches
            - Testing considerations'''
        })

    # Adjust based on complexity
    if question_analysis['complexity'] == 'complex':
        strategy['temperature'] = min(strategy['temperature'] + 0.1, 0.5)

    # Apply feedback if available
    if feedback_data:
        # Implement feedback-based adjustments
        pass

    return strategy

# Usage
questions = extract_questions_from_file("data/mixed_questions.xlsx", "Question", "Answer")
adaptive_templates, strategies = adaptive_template_generation(questions[:10])

print("Adaptive generation strategies:")
for q_id, strategy in strategies.items():
    print(f"  {q_id}: {strategy['model']} (temp: {strategy['temperature']})")
```

## Advanced Benchmark Workflows

### Multi-Dimensional Evaluation

```python
from karenina.benchmark.runner import run_benchmark
import concurrent.futures

def multi_dimensional_benchmark(questions_dict: dict, model_configs: list) -> dict:
    """Run benchmark across multiple dimensions: models, temperatures, prompts."""

    results = {}

    # Generate templates once
    print("Generating answer templates...")
    templates = generate_answer_templates_from_questions_file("questions.py")

    # Test different system prompts
    system_prompts = {
        'neutral': "Answer the question accurately and concisely.",
        'detailed': "Provide a comprehensive, detailed answer with examples and context.",
        'cautious': "Answer carefully, acknowledging any uncertainty or limitations.",
        'confident': "Answer with confidence and authority, providing definitive information."
    }

    # Run evaluations for each configuration
    for model_config in model_configs:
        model_key = f"{model_config['provider']}_{model_config['model']}"
        results[model_key] = {}

        for temp in [0.0, 0.3, 0.7]:
            results[model_key][f'temp_{temp}'] = {}

            for prompt_name, system_prompt in system_prompts.items():
                print(f"Testing {model_key} with temp={temp}, prompt={prompt_name}")

                # Collect responses
                responses_dict = collect_responses_with_config(
                    questions_dict,
                    model_config,
                    temperature=temp,
                    system_prompt=system_prompt
                )

                # Run benchmark
                benchmark_results = run_benchmark(questions_dict, responses_dict, templates)

                # Analyze results
                analysis = analyze_benchmark_results(benchmark_results)

                results[model_key][f'temp_{temp}'][prompt_name] = {
                    'benchmark_results': benchmark_results,
                    'analysis': analysis,
                    'config': {
                        'temperature': temp,
                        'system_prompt': prompt_name,
                        'model': model_config['model'],
                        'provider': model_config['provider']
                    }
                }

    return results

def collect_responses_with_config(questions_dict: dict, model_config: dict,
                                 temperature: float, system_prompt: str) -> dict:
    """Collect responses with specific configuration."""

    responses = {}

    for q_id, question in questions_dict.items():
        try:
            response = call_model(
                model=model_config['model'],
                provider=model_config['provider'],
                message=question,
                temperature=temperature,
                system_message=system_prompt
            )
            responses[q_id] = response.message

        except Exception as e:
            print(f"Error collecting response for {q_id}: {e}")
            responses[q_id] = ""

    return responses

def analyze_dimensional_results(results: dict) -> dict:
    """Analyze results across multiple dimensions."""

    analysis = {
        'temperature_effects': {},
        'prompt_effects': {},
        'model_comparison': {},
        'best_configurations': {}
    }

    # Analyze temperature effects
    for model_key, model_results in results.items():
        temp_scores = {}

        for temp_key, temp_results in model_results.items():
            # Average scores across prompts
            scores = []
            for prompt_results in temp_results.values():
                if 'analysis' in prompt_results:
                    # Extract relevant metrics
                    avg_confidence = prompt_results['analysis'].get('avg_confidence')
                    if avg_confidence:
                        scores.append(avg_confidence)

            if scores:
                temp_scores[temp_key] = statistics.mean(scores)

        analysis['temperature_effects'][model_key] = temp_scores

    # Analyze prompt effects
    for model_key, model_results in results.items():
        prompt_scores = {}

        for temp_key, temp_results in model_results.items():
            for prompt_name, prompt_results in temp_results.items():
                if prompt_name not in prompt_scores:
                    prompt_scores[prompt_name] = []

                if 'analysis' in prompt_results:
                    avg_confidence = prompt_results['analysis'].get('avg_confidence')
                    if avg_confidence:
                        prompt_scores[prompt_name].append(avg_confidence)

        # Average across temperatures
        for prompt_name in prompt_scores:
            if prompt_scores[prompt_name]:
                prompt_scores[prompt_name] = statistics.mean(prompt_scores[prompt_name])

        analysis['prompt_effects'][model_key] = prompt_scores

    return analysis

# Usage
model_configs = [
    {'model': 'gpt-4', 'provider': 'openai'},
    {'model': 'claude-3-sonnet', 'provider': 'anthropic'},
    {'model': 'gemini-2.0-flash', 'provider': 'google_genai'}
]

questions_dict = {"q1": "What is machine learning?", "q2": "How does neural network work?"}

dimensional_results = multi_dimensional_benchmark(questions_dict, model_configs)
dimensional_analysis = analyze_dimensional_results(dimensional_results)

print("Temperature effects:")
for model, temps in dimensional_analysis['temperature_effects'].items():
    print(f"  {model}: {temps}")

print("Prompt effects:")
for model, prompts in dimensional_analysis['prompt_effects'].items():
    print(f"  {model}: {prompts}")
```

### Continuous Evaluation Pipeline

```python
import time
import threading
from datetime import datetime
from pathlib import Path

class ContinuousEvaluator:
    """Continuous evaluation system for ongoing model assessment."""

    def __init__(self, config: dict):
        self.config = config
        self.results_history = []
        self.is_running = False
        self.evaluation_thread = None

    def start_continuous_evaluation(self):
        """Start continuous evaluation process."""

        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._evaluation_loop)
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()

        print("Continuous evaluation started")

    def stop_continuous_evaluation(self):
        """Stop continuous evaluation process."""

        self.is_running = False
        if self.evaluation_thread:
            self.evaluation_thread.join()

        print("Continuous evaluation stopped")

    def _evaluation_loop(self):
        """Main evaluation loop."""

        while self.is_running:
            try:
                # Run evaluation cycle
                results = self._run_evaluation_cycle()

                # Store results
                self.results_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                })

                # Save to disk
                self._save_results()

                # Check for performance changes
                self._analyze_performance_trends()

                # Wait for next cycle
                time.sleep(self.config.get('cycle_interval', 3600))  # Default 1 hour

            except Exception as e:
                print(f"Error in evaluation cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retry

    def _run_evaluation_cycle(self) -> dict:
        """Run a single evaluation cycle."""

        print(f"Running evaluation cycle at {datetime.now()}")

        # Load latest questions
        questions_dict = self._load_questions()

        # Generate/load templates
        templates = self._load_templates()

        # Collect fresh responses
        responses_dict = self._collect_fresh_responses(questions_dict)

        # Run benchmark
        results = run_benchmark(questions_dict, responses_dict, templates)

        # Analyze results
        analysis = analyze_benchmark_results(results)

        return {
            'benchmark_results': results,
            'analysis': analysis,
            'question_count': len(questions_dict),
            'response_count': len(responses_dict)
        }

    def _load_questions(self) -> dict:
        """Load questions for evaluation."""

        questions_file = self.config.get('questions_file', 'questions.py')
        questions = read_questions_from_file(questions_file)

        # Sample questions if too many
        max_questions = self.config.get('max_questions_per_cycle', 50)
        if len(questions) > max_questions:
            import random
            questions = random.sample(questions, max_questions)

        return {q.id: q.question for q in questions}

    def _load_templates(self) -> dict:
        """Load or generate answer templates."""

        cache_file = self.config.get('templates_cache', 'templates_cache.json')

        if Path(cache_file).exists():
            return load_answer_templates_from_json(cache_file)
        else:
            # Generate fresh templates
            templates = generate_answer_templates_from_questions_file(
                self.config.get('questions_file', 'questions.py')
            )

            # Save to cache
            templates, code_blocks = generate_answer_templates_from_questions_file(
                self.config.get('questions_file', 'questions.py'),
                return_blocks=True
            )

            with open(cache_file, 'w') as f:
                json.dump(code_blocks, f, indent=2)

            return templates

    def _collect_fresh_responses(self, questions_dict: dict) -> dict:
        """Collect fresh responses from configured models."""

        responses = {}
        model_config = self.config['model']

        for q_id, question in questions_dict.items():
            try:
                response = call_model(
                    model=model_config['model'],
                    provider=model_config['provider'],
                    message=question,
                    temperature=model_config.get('temperature', 0.3)
                )
                responses[q_id] = response.message

            except Exception as e:
                print(f"Error getting response for {q_id}: {e}")
                responses[q_id] = ""

        return responses

    def _save_results(self):
        """Save evaluation results to disk."""

        results_file = self.config.get('results_file', 'continuous_results.json')

        with open(results_file, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)

    def _analyze_performance_trends(self):
        """Analyze performance trends over time."""

        if len(self.results_history) < 2:
            return

        # Get last two results
        latest = self.results_history[-1]
        previous = self.results_history[-2]

        # Compare key metrics
        latest_confidence = latest['results']['analysis'].get('avg_confidence')
        previous_confidence = previous['results']['analysis'].get('avg_confidence')

        if latest_confidence and previous_confidence:
            change = latest_confidence - previous_confidence

            if abs(change) > 0.1:  # Significant change threshold
                print(f"Performance change detected: {change:+.3f} confidence")

                if change < -0.1:
                    print("⚠️  Performance degradation detected!")
                elif change > 0.1:
                    print("✅ Performance improvement detected!")

    def get_performance_summary(self) -> dict:
        """Get summary of performance over time."""

        if not self.results_history:
            return {"error": "No evaluation history available"}

        confidences = []
        timestamps = []

        for entry in self.results_history:
            confidence = entry['results']['analysis'].get('avg_confidence')
            if confidence:
                confidences.append(confidence)
                timestamps.append(entry['timestamp'])

        if not confidences:
            return {"error": "No confidence data available"}

        return {
            'total_evaluations': len(self.results_history),
            'latest_confidence': confidences[-1] if confidences else None,
            'avg_confidence': statistics.mean(confidences),
            'confidence_trend': confidences[-5:] if len(confidences) >= 5 else confidences,
            'evaluation_period': {
                'start': timestamps[0] if timestamps else None,
                'end': timestamps[-1] if timestamps else None
            }
        }

# Usage
evaluator_config = {
    'model': {
        'model': 'gpt-4',
        'provider': 'openai',
        'temperature': 0.3
    },
    'questions_file': 'benchmark_questions.py',
    'templates_cache': 'templates_cache.json',
    'results_file': 'continuous_results.json',
    'cycle_interval': 1800,  # 30 minutes
    'max_questions_per_cycle': 25
}

evaluator = ContinuousEvaluator(evaluator_config)

# Start continuous evaluation
evaluator.start_continuous_evaluation()

# Let it run for a while...
time.sleep(10)

# Get summary
summary = evaluator.get_performance_summary()
print("Performance summary:", summary)

# Stop evaluation
evaluator.stop_continuous_evaluation()
```

## Integration Patterns

### API Integration Pipeline

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid

app = FastAPI()

class EvaluationRequest(BaseModel):
    questions_file: str
    model_config: dict
    evaluation_id: str = None

class EvaluationStatus(BaseModel):
    evaluation_id: str
    status: str
    progress: float
    results: dict = None

# Global storage for evaluation status
evaluation_store = {}

@app.post("/start_evaluation", response_model=EvaluationStatus)
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start asynchronous evaluation process."""

    evaluation_id = request.evaluation_id or str(uuid.uuid4())

    # Initialize status
    evaluation_store[evaluation_id] = {
        'status': 'starting',
        'progress': 0.0,
        'results': None
    }

    # Start background evaluation
    background_tasks.add_task(
        run_background_evaluation,
        evaluation_id,
        request.questions_file,
        request.model_config
    )

    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status='starting',
        progress=0.0
    )

@app.get("/evaluation_status/{evaluation_id}", response_model=EvaluationStatus)
async def get_evaluation_status(evaluation_id: str):
    """Get evaluation status and results."""

    if evaluation_id not in evaluation_store:
        return EvaluationStatus(
            evaluation_id=evaluation_id,
            status='not_found',
            progress=0.0
        )

    status_data = evaluation_store[evaluation_id]

    return EvaluationStatus(
        evaluation_id=evaluation_id,
        status=status_data['status'],
        progress=status_data['progress'],
        results=status_data['results']
    )

async def run_background_evaluation(evaluation_id: str, questions_file: str, model_config: dict):
    """Run evaluation in background with progress tracking."""

    try:
        # Update status
        evaluation_store[evaluation_id]['status'] = 'loading_questions'
        evaluation_store[evaluation_id]['progress'] = 0.1

        # Load questions
        questions = read_questions_from_file(questions_file)
        questions_dict = {q.id: q.question for q in questions}

        # Generate templates
        evaluation_store[evaluation_id]['status'] = 'generating_templates'
        evaluation_store[evaluation_id]['progress'] = 0.3

        templates = generate_answer_templates_from_questions_file(questions_file)

        # Collect responses
        evaluation_store[evaluation_id]['status'] = 'collecting_responses'
        evaluation_store[evaluation_id]['progress'] = 0.5

        responses_dict = {}
        total_questions = len(questions_dict)

        for i, (q_id, question) in enumerate(questions_dict.items()):
            try:
                response = call_model(
                    model=model_config['model'],
                    provider=model_config['provider'],
                    message=question,
                    temperature=model_config.get('temperature', 0.3)
                )
                responses_dict[q_id] = response.message

                # Update progress
                progress = 0.5 + (0.3 * (i + 1) / total_questions)
                evaluation_store[evaluation_id]['progress'] = progress

            except Exception as e:
                responses_dict[q_id] = f"Error: {str(e)}"

        # Run benchmark
        evaluation_store[evaluation_id]['status'] = 'running_benchmark'
        evaluation_store[evaluation_id]['progress'] = 0.8

        results = run_benchmark(questions_dict, responses_dict, templates)
        analysis = analyze_benchmark_results(results)

        # Complete evaluation
        evaluation_store[evaluation_id]['status'] = 'completed'
        evaluation_store[evaluation_id]['progress'] = 1.0
        evaluation_store[evaluation_id]['results'] = {
            'benchmark_results': {k: v.model_dump() for k, v in results.items()},
            'analysis': analysis,
            'model_config': model_config
        }

    except Exception as e:
        evaluation_store[evaluation_id]['status'] = 'failed'
        evaluation_store[evaluation_id]['results'] = {'error': str(e)}

# Example client usage
import httpx

async def run_api_evaluation():
    """Example of using the API for evaluation."""

    async with httpx.AsyncClient() as client:
        # Start evaluation
        response = await client.post("http://localhost:8000/start_evaluation", json={
            "questions_file": "benchmark_questions.py",
            "model_config": {
                "model": "gpt-4",
                "provider": "openai",
                "temperature": 0.3
            }
        })

        evaluation_data = response.json()
        evaluation_id = evaluation_data['evaluation_id']

        print(f"Started evaluation: {evaluation_id}")

        # Poll for completion
        while True:
            status_response = await client.get(f"http://localhost:8000/evaluation_status/{evaluation_id}")
            status_data = status_response.json()

            print(f"Status: {status_data['status']} ({status_data['progress']:.1%})")

            if status_data['status'] in ['completed', 'failed']:
                break

            await asyncio.sleep(5)

        if status_data['status'] == 'completed':
            print("Evaluation completed!")
            print("Results:", status_data['results']['analysis'])
        else:
            print("Evaluation failed:", status_data['results'])

# Usage
# uvicorn main:app --reload
# Then run the client
# asyncio.run(run_api_evaluation())
```
