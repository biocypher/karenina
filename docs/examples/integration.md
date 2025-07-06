# Integration Examples

Examples showing how to integrate Karenina with other systems and frameworks.

## Web Framework Integration

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import json
import uuid
from pathlib import Path

from karenina.questions.extractor import get_file_preview, extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark
from karenina.llm.interface import call_model

app = FastAPI(title="Karenina Benchmarking API")

# Request/Response Models
class FilePreviewResponse(BaseModel):
    success: bool
    columns: list = None
    preview_data: list = None
    total_rows: int = None
    error: str = None

class QuestionExtractionRequest(BaseModel):
    question_column: str
    answer_column: str
    sheet_name: str = None

class TemplateGenerationRequest(BaseModel):
    model: str = "gemini-2.0-flash"
    provider: str = "google_genai"

class BenchmarkRequest(BaseModel):
    target_model: str
    target_provider: str
    temperature: float = 0.3

# Global storage for job tracking
job_storage = {}

@app.post("/upload_file", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    """Upload and preview a question file."""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_path = tmp_file.name

    try:
        # Get file preview
        preview = get_file_preview(temp_path, max_rows=10)

        if preview["success"]:
            # Store file for later use
            file_id = str(uuid.uuid4())
            job_storage[file_id] = {
                'file_path': temp_path,
                'original_name': file.filename,
                'preview': preview
            }

            return {
                "file_id": file_id,
                "preview": preview
            }
        else:
            return {"error": preview["error"]}

    except Exception as e:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/extract_questions/{file_id}")
async def extract_questions(
    file_id: str,
    extraction_request: QuestionExtractionRequest,
    background_tasks: BackgroundTasks
):
    """Extract questions from uploaded file."""

    if file_id not in job_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = job_storage[file_id]

    # Start background extraction
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {
        'status': 'starting',
        'progress': 0.0,
        'type': 'extraction'
    }

    background_tasks.add_task(
        extract_questions_background,
        job_id,
        file_info['file_path'],
        extraction_request
    )

    return {"job_id": job_id, "status": "started"}

async def extract_questions_background(job_id: str, file_path: str, request: QuestionExtractionRequest):
    """Background task for question extraction."""

    try:
        job_storage[job_id]['status'] = 'extracting'
        job_storage[job_id]['progress'] = 0.3

        # Generate unique output file
        output_file = f"questions_{job_id}.py"

        # Extract questions
        questions_json = extract_and_generate_questions(
            file_path=file_path,
            output_path=output_file,
            question_column=request.question_column,
            answer_column=request.answer_column,
            sheet_name=request.sheet_name,
            return_json=True
        )

        job_storage[job_id].update({
            'status': 'completed',
            'progress': 1.0,
            'questions_file': output_file,
            'questions_json': questions_json,
            'question_count': len(questions_json)
        })

    except Exception as e:
        job_storage[job_id].update({
            'status': 'failed',
            'error': str(e)
        })

@app.post("/generate_templates/{job_id}")
async def generate_templates(
    job_id: str,
    template_request: TemplateGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate answer templates for extracted questions."""

    if job_id not in job_storage or job_storage[job_id].get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Questions not ready")

    template_job_id = str(uuid.uuid4())
    job_storage[template_job_id] = {
        'status': 'starting',
        'progress': 0.0,
        'type': 'template_generation'
    }

    background_tasks.add_task(
        generate_templates_background,
        template_job_id,
        job_storage[job_id]['questions_file'],
        template_request
    )

    return {"template_job_id": template_job_id, "status": "started"}

async def generate_templates_background(job_id: str, questions_file: str, request: TemplateGenerationRequest):
    """Background task for template generation."""

    try:
        job_storage[job_id]['status'] = 'generating'
        job_storage[job_id]['progress'] = 0.2

        # Generate templates
        templates, code_blocks = generate_answer_templates_from_questions_file(
            questions_file,
            model=request.model,
            model_provider=request.provider,
            return_blocks=True
        )

        # Save templates
        templates_file = f"templates_{job_id}.json"
        with open(templates_file, 'w') as f:
            json.dump(code_blocks, f, indent=2)

        job_storage[job_id].update({
            'status': 'completed',
            'progress': 1.0,
            'templates_file': templates_file,
            'template_count': len(templates),
            'questions_file': questions_file
        })

    except Exception as e:
        job_storage[job_id].update({
            'status': 'failed',
            'error': str(e)
        })

@app.post("/run_benchmark/{template_job_id}")
async def run_benchmark_endpoint(
    template_job_id: str,
    benchmark_request: BenchmarkRequest,
    background_tasks: BackgroundTasks
):
    """Run benchmark evaluation."""

    if template_job_id not in job_storage or job_storage[template_job_id].get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Templates not ready")

    benchmark_job_id = str(uuid.uuid4())
    job_storage[benchmark_job_id] = {
        'status': 'starting',
        'progress': 0.0,
        'type': 'benchmark'
    }

    background_tasks.add_task(
        run_benchmark_background,
        benchmark_job_id,
        job_storage[template_job_id],
        benchmark_request
    )

    return {"benchmark_job_id": benchmark_job_id, "status": "started"}

async def run_benchmark_background(job_id: str, template_info: dict, request: BenchmarkRequest):
    """Background task for benchmark execution."""

    try:
        job_storage[job_id]['status'] = 'loading'
        job_storage[job_id]['progress'] = 0.1

        # Load questions and templates
        from karenina.questions.reader import read_questions_from_file
        from karenina.answers.generator import load_answer_templates_from_json

        questions = read_questions_from_file(template_info['questions_file'])
        questions_dict = {q.id: q.question for q in questions}
        templates = load_answer_templates_from_json(template_info['templates_file'])

        # Collect responses
        job_storage[job_id]['status'] = 'collecting_responses'
        job_storage[job_id]['progress'] = 0.3

        responses_dict = {}
        total_questions = len(questions_dict)

        for i, (q_id, question) in enumerate(questions_dict.items()):
            try:
                response = call_model(
                    model=request.target_model,
                    provider=request.target_provider,
                    message=question,
                    temperature=request.temperature
                )
                responses_dict[q_id] = response.message

                # Update progress
                progress = 0.3 + (0.5 * (i + 1) / total_questions)
                job_storage[job_id]['progress'] = progress

            except Exception as e:
                responses_dict[q_id] = f"Error: {str(e)}"

        # Run benchmark
        job_storage[job_id]['status'] = 'evaluating'
        job_storage[job_id]['progress'] = 0.8

        results = run_benchmark(questions_dict, responses_dict, templates)

        # Serialize results
        serialized_results = {}
        for q_id, result in results.items():
            serialized_results[q_id] = result.model_dump()

        job_storage[job_id].update({
            'status': 'completed',
            'progress': 1.0,
            'results': serialized_results,
            'summary': {
                'total_questions': len(results),
                'model': request.target_model,
                'provider': request.target_provider,
                'temperature': request.temperature
            }
        })

    except Exception as e:
        job_storage[job_id].update({
            'status': 'failed',
            'error': str(e)
        })

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of any background job."""

    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_storage[job_id]

@app.get("/results/{benchmark_job_id}")
async def get_benchmark_results(benchmark_job_id: str):
    """Get detailed benchmark results."""

    if benchmark_job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = job_storage[benchmark_job_id]

    if job_info.get('status') != 'completed':
        raise HTTPException(status_code=400, detail="Benchmark not completed")

    return {
        'summary': job_info['summary'],
        'results': job_info['results']
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "karenina-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Integration

```python
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import os
import json
from datetime import datetime

from karenina.questions.extractor import get_file_preview, extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Simple in-memory storage (use database in production)
uploaded_files = {}
extraction_results = {}

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and preview question file."""

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    try:
        # Get preview
        preview = get_file_preview(file_path, max_rows=5)

        if preview['success']:
            file_id = f"file_{len(uploaded_files)}"
            uploaded_files[file_id] = {
                'path': file_path,
                'original_name': filename,
                'upload_time': datetime.now().isoformat(),
                'preview': preview
            }

            return jsonify({
                'file_id': file_id,
                'preview': preview
            })
        else:
            return jsonify({'error': preview['error']}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract/<file_id>', methods=['POST'])
def extract_questions(file_id):
    """Extract questions from uploaded file."""

    if file_id not in uploaded_files:
        return jsonify({'error': 'File not found'}), 404

    data = request.get_json()
    question_column = data.get('question_column', 'Question')
    answer_column = data.get('answer_column', 'Answer')
    sheet_name = data.get('sheet_name')

    file_info = uploaded_files[file_id]

    try:
        # Extract questions
        questions_json = extract_and_generate_questions(
            file_path=file_info['path'],
            output_path=f'questions_{file_id}.py',
            question_column=question_column,
            answer_column=answer_column,
            sheet_name=sheet_name,
            return_json=True
        )

        extraction_id = f"extraction_{len(extraction_results)}"
        extraction_results[extraction_id] = {
            'file_id': file_id,
            'questions_json': questions_json,
            'questions_file': f'questions_{file_id}.py',
            'extraction_time': datetime.now().isoformat(),
            'question_count': len(questions_json)
        }

        return jsonify({
            'extraction_id': extraction_id,
            'question_count': len(questions_json),
            'questions': list(questions_json.values())[:5]  # First 5 for preview
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_templates/<extraction_id>', methods=['POST'])
def generate_templates(extraction_id):
    """Generate answer templates."""

    if extraction_id not in extraction_results:
        return jsonify({'error': 'Extraction not found'}), 404

    data = request.get_json()
    model = data.get('model', 'gemini-2.0-flash')
    provider = data.get('provider', 'google_genai')

    extraction_info = extraction_results[extraction_id]

    try:
        # Generate templates
        templates, code_blocks = generate_answer_templates_from_questions_file(
            extraction_info['questions_file'],
            model=model,
            model_provider=provider,
            return_blocks=True
        )

        # Save templates
        templates_file = f'templates_{extraction_id}.json'
        with open(templates_file, 'w') as f:
            json.dump(code_blocks, f, indent=2)

        # Update extraction results
        extraction_results[extraction_id].update({
            'templates_file': templates_file,
            'template_count': len(templates),
            'generation_model': f"{provider}:{model}",
            'generation_time': datetime.now().isoformat()
        })

        return jsonify({
            'template_count': len(templates),
            'templates_file': templates_file,
            'sample_template': list(code_blocks.values())[0] if code_blocks else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/benchmark/<extraction_id>', methods=['POST'])
def run_benchmark_endpoint(extraction_id):
    """Run benchmark evaluation."""

    if extraction_id not in extraction_results:
        return jsonify({'error': 'Extraction not found'}), 404

    extraction_info = extraction_results[extraction_id]

    if 'templates_file' not in extraction_info:
        return jsonify({'error': 'Templates not generated yet'}), 400

    data = request.get_json()
    target_model = data.get('model', 'gpt-3.5-turbo')
    target_provider = data.get('provider', 'openai')
    temperature = data.get('temperature', 0.3)

    try:
        # Load questions and templates
        from karenina.questions.reader import read_questions_from_file
        from karenina.answers.generator import load_answer_templates_from_json
        from karenina.benchmark.runner import run_benchmark
        from karenina.llm.interface import call_model

        questions = read_questions_from_file(extraction_info['questions_file'])
        questions_dict = {q.id: q.question for q in questions}
        templates = load_answer_templates_from_json(extraction_info['templates_file'])

        # Collect responses (limit to first 10 for demo)
        limited_questions = dict(list(questions_dict.items())[:10])
        responses_dict = {}

        for q_id, question in limited_questions.items():
            try:
                response = call_model(
                    model=target_model,
                    provider=target_provider,
                    message=question,
                    temperature=temperature
                )
                responses_dict[q_id] = response.message
            except Exception as e:
                responses_dict[q_id] = f"Error: {str(e)}"

        # Run benchmark
        results = run_benchmark(limited_questions, responses_dict, templates)

        # Serialize results
        serialized_results = []
        for q_id, result in results.items():
            serialized_results.append({
                'question_id': q_id,
                'question': limited_questions[q_id],
                'response': responses_dict[q_id],
                'evaluation': result.model_dump()
            })

        return jsonify({
            'benchmark_results': serialized_results,
            'summary': {
                'total_evaluated': len(results),
                'target_model': f"{target_provider}:{target_model}",
                'temperature': temperature,
                'evaluation_time': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_templates/<extraction_id>')
def download_templates(extraction_id):
    """Download generated templates file."""

    if extraction_id not in extraction_results:
        return jsonify({'error': 'Extraction not found'}), 404

    extraction_info = extraction_results[extraction_id]

    if 'templates_file' not in extraction_info:
        return jsonify({'error': 'Templates not generated yet'}), 400

    return send_file(
        extraction_info['templates_file'],
        as_attachment=True,
        download_name=f'answer_templates_{extraction_id}.json'
    )

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Database Integration

### SQLAlchemy Models

```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class QuestionDataset(Base):
    """Table for storing question datasets."""

    __tablename__ = 'question_datasets'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_path = Column(String(500))
    question_column = Column(String(100))
    answer_column = Column(String(100))
    sheet_name = Column(String(100))
    total_questions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    questions = relationship("Question", back_populates="dataset")
    evaluations = relationship("Evaluation", back_populates="dataset")

class Question(Base):
    """Table for storing individual questions."""

    __tablename__ = 'questions'

    id = Column(String(32), primary_key=True)  # MD5 hash
    dataset_id = Column(Integer, ForeignKey('question_datasets.id'))
    question_text = Column(Text, nullable=False)
    raw_answer = Column(Text, nullable=False)
    tags = Column(JSON)  # Store as JSON array
    complexity_score = Column(Float)
    word_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship("QuestionDataset", back_populates="questions")
    templates = relationship("AnswerTemplate", back_populates="question")
    evaluations = relationship("EvaluationResult", back_populates="question")

class AnswerTemplate(Base):
    """Table for storing generated answer templates."""

    __tablename__ = 'answer_templates'

    id = Column(Integer, primary_key=True)
    question_id = Column(String(32), ForeignKey('questions.id'))
    template_code = Column(Text, nullable=False)
    generation_model = Column(String(100))
    generation_provider = Column(String(50))
    field_count = Column(Integer)
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    question = relationship("Question", back_populates="templates")
    evaluations = relationship("EvaluationResult", back_populates="template")

class Evaluation(Base):
    """Table for storing evaluation runs."""

    __tablename__ = 'evaluations'

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('question_datasets.id'))
    name = Column(String(255))
    target_model = Column(String(100))
    target_provider = Column(String(50))
    temperature = Column(Float)
    total_questions = Column(Integer)
    completed_questions = Column(Integer)
    avg_confidence = Column(Float)
    status = Column(String(50))  # pending, running, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    dataset = relationship("QuestionDataset", back_populates="evaluations")
    results = relationship("EvaluationResult", back_populates="evaluation")

class EvaluationResult(Base):
    """Table for storing individual evaluation results."""

    __tablename__ = 'evaluation_results'

    id = Column(Integer, primary_key=True)
    evaluation_id = Column(Integer, ForeignKey('evaluations.id'))
    question_id = Column(String(32), ForeignKey('questions.id'))
    template_id = Column(Integer, ForeignKey('answer_templates.id'))
    model_response = Column(Text)
    structured_result = Column(JSON)  # Store validated Pydantic result
    confidence_score = Column(Float)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    evaluation = relationship("Evaluation", back_populates="results")
    question = relationship("Question", back_populates="evaluations")
    template = relationship("AnswerTemplate", back_populates="evaluations")

# Database service class
class KareninaDatabaseService:
    """Service for database operations."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def create_dataset(self, name: str, file_path: str, question_column: str,
                      answer_column: str, sheet_name: str = None) -> QuestionDataset:
        """Create a new question dataset."""

        dataset = QuestionDataset(
            name=name,
            file_path=file_path,
            question_column=question_column,
            answer_column=answer_column,
            sheet_name=sheet_name
        )

        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)

        return dataset

    def store_questions(self, dataset_id: int, questions: list) -> None:
        """Store questions for a dataset."""

        question_objects = []
        for q in questions:
            question_obj = Question(
                id=q.id,
                dataset_id=dataset_id,
                question_text=q.question,
                raw_answer=q.raw_answer,
                tags=q.tags,
                word_count=len(q.question.split())
            )
            question_objects.append(question_obj)

        self.session.add_all(question_objects)

        # Update dataset total
        dataset = self.session.query(QuestionDataset).get(dataset_id)
        dataset.total_questions = len(question_objects)

        self.session.commit()

    def store_answer_template(self, question_id: str, template_code: str,
                            model: str, provider: str) -> AnswerTemplate:
        """Store an answer template."""

        # Count fields in template
        field_count = template_code.count('Field(')

        template = AnswerTemplate(
            question_id=question_id,
            template_code=template_code,
            generation_model=model,
            generation_provider=provider,
            field_count=field_count
        )

        self.session.add(template)
        self.session.commit()
        self.session.refresh(template)

        return template

    def create_evaluation(self, dataset_id: int, name: str, target_model: str,
                         target_provider: str, temperature: float) -> Evaluation:
        """Create a new evaluation run."""

        dataset = self.session.query(QuestionDataset).get(dataset_id)

        evaluation = Evaluation(
            dataset_id=dataset_id,
            name=name,
            target_model=target_model,
            target_provider=target_provider,
            temperature=temperature,
            total_questions=dataset.total_questions,
            completed_questions=0,
            status='pending'
        )

        self.session.add(evaluation)
        self.session.commit()
        self.session.refresh(evaluation)

        return evaluation

    def store_evaluation_result(self, evaluation_id: int, question_id: str,
                              template_id: int, response: str,
                              structured_result: dict, confidence: float = None) -> None:
        """Store an evaluation result."""

        result = EvaluationResult(
            evaluation_id=evaluation_id,
            question_id=question_id,
            template_id=template_id,
            model_response=response,
            structured_result=structured_result,
            confidence_score=confidence
        )

        self.session.add(result)

        # Update evaluation progress
        evaluation = self.session.query(Evaluation).get(evaluation_id)
        evaluation.completed_questions += 1

        if evaluation.completed_questions >= evaluation.total_questions:
            evaluation.status = 'completed'
            evaluation.completed_at = datetime.utcnow()

            # Calculate average confidence
            avg_confidence = self.session.query(EvaluationResult.confidence_score).filter(
                EvaluationResult.evaluation_id == evaluation_id,
                EvaluationResult.confidence_score.isnot(None)
            ).all()

            if avg_confidence:
                evaluation.avg_confidence = sum(c[0] for c in avg_confidence) / len(avg_confidence)

        self.session.commit()

    def get_evaluation_summary(self, evaluation_id: int) -> dict:
        """Get evaluation summary with statistics."""

        evaluation = self.session.query(Evaluation).get(evaluation_id)
        if not evaluation:
            return None

        results = self.session.query(EvaluationResult).filter(
            EvaluationResult.evaluation_id == evaluation_id
        ).all()

        confidence_scores = [r.confidence_score for r in results if r.confidence_score]

        return {
            'evaluation_id': evaluation_id,
            'name': evaluation.name,
            'status': evaluation.status,
            'target_model': f"{evaluation.target_provider}:{evaluation.target_model}",
            'temperature': evaluation.temperature,
            'total_questions': evaluation.total_questions,
            'completed_questions': evaluation.completed_questions,
            'avg_confidence': evaluation.avg_confidence,
            'confidence_distribution': {
                'min': min(confidence_scores) if confidence_scores else None,
                'max': max(confidence_scores) if confidence_scores else None,
                'count': len(confidence_scores)
            },
            'started_at': evaluation.started_at.isoformat(),
            'completed_at': evaluation.completed_at.isoformat() if evaluation.completed_at else None
        }

# Usage example
def database_integration_example():
    """Example of using the database service."""

    # Initialize database
    db_service = KareninaDatabaseService("sqlite:///karenina.db")

    # Create dataset
    dataset = db_service.create_dataset(
        name="Math Questions",
        file_path="data/math_questions.xlsx",
        question_column="Question",
        answer_column="Answer"
    )

    # Extract and store questions
    from karenina.questions.extractor import extract_questions_from_file
    questions = extract_questions_from_file(
        dataset.file_path,
        dataset.question_column,
        dataset.answer_column
    )

    db_service.store_questions(dataset.id, questions)

    # Generate and store templates
    from karenina.answers.generator import generate_answer_template

    for question in questions[:5]:  # First 5 for demo
        template_code = generate_answer_template(
            question.question,
            question.model_dump_json()
        )

        db_service.store_answer_template(
            question.id,
            template_code,
            "gemini-2.0-flash",
            "google_genai"
        )

    # Create evaluation
    evaluation = db_service.create_evaluation(
        dataset.id,
        "GPT-4 Evaluation",
        "gpt-4",
        "openai",
        0.3
    )

    print(f"Created evaluation: {evaluation.id}")

    # Get summary
    summary = db_service.get_evaluation_summary(evaluation.id)
    print("Evaluation summary:", summary)

if __name__ == "__main__":
    database_integration_example()
```

## Message Queue Integration

### Celery Integration

```python
from celery import Celery
from celery.result import AsyncResult
import json
import tempfile
from datetime import datetime

from karenina.questions.extractor import extract_and_generate_questions
from karenina.answers.generator import generate_answer_templates_from_questions_file
from karenina.benchmark.runner import run_benchmark
from karenina.llm.interface import call_model

# Celery configuration
celery_app = Celery(
    'karenina_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
)

@celery_app.task(bind=True)
def extract_questions_task(self, file_path: str, question_column: str,
                          answer_column: str, sheet_name: str = None):
    """Celery task for question extraction."""

    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Starting extraction'})

        # Extract questions
        questions_json = extract_and_generate_questions(
            file_path=file_path,
            output_path=f'questions_{self.request.id}.py',
            question_column=question_column,
            answer_column=answer_column,
            sheet_name=sheet_name,
            return_json=True
        )

        self.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Finalizing'})

        result = {
            'questions_file': f'questions_{self.request.id}.py',
            'questions_json': questions_json,
            'question_count': len(questions_json),
            'extraction_time': datetime.utcnow().isoformat()
        }

        self.update_state(state='SUCCESS', meta={'progress': 100, 'result': result})
        return result

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def generate_templates_task(self, questions_file: str, model: str = "gemini-2.0-flash",
                           provider: str = "google_genai"):
    """Celery task for template generation."""

    try:
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading questions'})

        # Generate templates with progress updates
        from karenina.questions.reader import read_questions_from_file
        questions = read_questions_from_file(questions_file)

        templates = {}
        code_blocks = {}

        total_questions = len(questions)

        for i, question in enumerate(questions):
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': 10 + (70 * i / total_questions),
                    'status': f'Generating template {i+1}/{total_questions}'
                }
            )

            template_code = generate_answer_template(
                question.question,
                question.model_dump_json(),
                model=model,
                model_provider=provider
            )

            from karenina.utils.code_parser import extract_and_combine_codeblocks
            code_block = extract_and_combine_codeblocks(template_code)

            # Execute template
            local_ns = {}
            exec(code_block, globals(), local_ns)

            templates[question.id] = local_ns["Answer"]
            code_blocks[question.id] = code_block

        # Save templates
        templates_file = f'templates_{self.request.id}.json'
        with open(templates_file, 'w') as f:
            json.dump(code_blocks, f, indent=2)

        result = {
            'templates_file': templates_file,
            'template_count': len(templates),
            'generation_model': f"{provider}:{model}",
            'generation_time': datetime.utcnow().isoformat()
        }

        self.update_state(state='SUCCESS', meta={'progress': 100, 'result': result})
        return result

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

@celery_app.task(bind=True)
def run_benchmark_task(self, questions_file: str, templates_file: str,
                      target_model: str, target_provider: str, temperature: float = 0.3):
    """Celery task for benchmark execution."""

    try:
        self.update_state(state='PROGRESS', meta={'progress': 5, 'status': 'Loading data'})

        # Load questions and templates
        from karenina.questions.reader import read_questions_from_file
        from karenina.answers.generator import load_answer_templates_from_json

        questions = read_questions_from_file(questions_file)
        questions_dict = {q.id: q.question for q in questions}
        templates = load_answer_templates_from_json(templates_file)

        self.update_state(state='PROGRESS', meta={'progress': 15, 'status': 'Collecting responses'})

        # Collect responses with progress
        responses_dict = {}
        total_questions = len(questions_dict)

        for i, (q_id, question) in enumerate(questions_dict.items()):
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': 15 + (60 * i / total_questions),
                    'status': f'Collecting response {i+1}/{total_questions}'
                }
            )

            try:
                response = call_model(
                    model=target_model,
                    provider=target_provider,
                    message=question,
                    temperature=temperature
                )
                responses_dict[q_id] = response.message
            except Exception as e:
                responses_dict[q_id] = f"Error: {str(e)}"

        self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Running evaluation'})

        # Run benchmark
        results = run_benchmark(questions_dict, responses_dict, templates)

        # Serialize results
        serialized_results = {}
        for q_id, result in results.items():
            serialized_results[q_id] = result.model_dump()

        result = {
            'benchmark_results': serialized_results,
            'summary': {
                'total_questions': len(results),
                'target_model': f"{target_provider}:{target_model}",
                'temperature': temperature,
                'evaluation_time': datetime.utcnow().isoformat()
            }
        }

        self.update_state(state='SUCCESS', meta={'progress': 100, 'result': result})
        return result

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

# Task monitoring and management
class TaskManager:
    """Manager for Celery tasks."""

    @staticmethod
    def start_full_pipeline(file_path: str, question_column: str, answer_column: str,
                           template_model: str = "gemini-2.0-flash",
                           template_provider: str = "google_genai",
                           target_model: str = "gpt-4",
                           target_provider: str = "openai",
                           temperature: float = 0.3) -> str:
        """Start complete pipeline as chained tasks."""

        from celery import chain

        # Create task chain
        pipeline = chain(
            extract_questions_task.s(file_path, question_column, answer_column),
            generate_templates_task.s(template_model, template_provider),
            run_benchmark_task.s(target_model, target_provider, temperature)
        )

        # Execute chain
        result = pipeline.apply_async()
        return result.id

    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Get status of any task."""

        result = AsyncResult(task_id, app=celery_app)

        if result.state == 'PENDING':
            return {'state': 'PENDING', 'progress': 0, 'status': 'Waiting to start'}
        elif result.state == 'PROGRESS':
            return {
                'state': 'PROGRESS',
                'progress': result.info.get('progress', 0),
                'status': result.info.get('status', 'Processing')
            }
        elif result.state == 'SUCCESS':
            return {
                'state': 'SUCCESS',
                'progress': 100,
                'result': result.result
            }
        elif result.state == 'FAILURE':
            return {
                'state': 'FAILURE',
                'error': str(result.info)
            }
        else:
            return {'state': result.state, 'info': str(result.info)}

    @staticmethod
    def cancel_task(task_id: str) -> bool:
        """Cancel a running task."""

        celery_app.control.revoke(task_id, terminate=True)
        return True

# Flask endpoint integration
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/start_pipeline', methods=['POST'])
def start_pipeline():
    """Start the complete benchmarking pipeline."""

    data = request.get_json()

    try:
        task_id = TaskManager.start_full_pipeline(
            file_path=data['file_path'],
            question_column=data['question_column'],
            answer_column=data['answer_column'],
            template_model=data.get('template_model', 'gemini-2.0-flash'),
            template_provider=data.get('template_provider', 'google_genai'),
            target_model=data.get('target_model', 'gpt-4'),
            target_provider=data.get('target_provider', 'openai'),
            temperature=data.get('temperature', 0.3)
        )

        return jsonify({'task_id': task_id, 'status': 'started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_status/<task_id>')
def get_task_status(task_id):
    """Get task status and progress."""

    status = TaskManager.get_task_status(task_id)
    return jsonify(status)

@app.route('/api/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running task."""

    success = TaskManager.cancel_task(task_id)
    return jsonify({'cancelled': success})

if __name__ == '__main__':
    # Start Celery worker: celery -A integration worker --loglevel=info
    # Start Flask app: python integration.py
    app.run(debug=True, port=5000)
```

## Cloud Integration

### AWS Lambda Integration

```python
import json
import boto3
import tempfile
import os
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """AWS Lambda handler for Karenina benchmarking."""

    try:
        # Parse event
        operation = event.get('operation', 'benchmark')

        if operation == 'extract_questions':
            return handle_question_extraction(event)
        elif operation == 'generate_templates':
            return handle_template_generation(event)
        elif operation == 'run_benchmark':
            return handle_benchmark_execution(event)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown operation: {operation}'})
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_question_extraction(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle question extraction in Lambda."""

    # Get S3 file
    s3_bucket = event['s3_bucket']
    s3_key = event['s3_key']
    question_column = event['question_column']
    answer_column = event['answer_column']

    # Download file from S3
    s3 = boto3.client('s3')

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        s3.download_fileobj(s3_bucket, s3_key, tmp_file)
        tmp_file_path = tmp_file.name

    try:
        from karenina.questions.extractor import extract_and_generate_questions

        # Extract questions
        questions_json = extract_and_generate_questions(
            file_path=tmp_file_path,
            output_path='',  # Not used when return_json=True
            question_column=question_column,
            answer_column=answer_column,
            return_json=True
        )

        # Upload results to S3
        output_key = f"extracted_questions/{event.get('request_id', 'default')}.json"
        s3.put_object(
            Bucket=s3_bucket,
            Key=output_key,
            Body=json.dumps(questions_json),
            ContentType='application/json'
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Questions extracted successfully',
                'question_count': len(questions_json),
                'output_s3_key': output_key
            })
        }

    finally:
        # Clean up
        os.unlink(tmp_file_path)

def handle_template_generation(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle template generation in Lambda."""

    # This would be too time-consuming for Lambda
    # Better to use Step Functions or ECS
    return {
        'statusCode': 400,
        'body': json.dumps({
            'error': 'Template generation too resource-intensive for Lambda. Use Step Functions or ECS.'
        })
    }

def handle_benchmark_execution(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle lightweight benchmark execution."""

    # For small datasets only
    max_questions = 10

    questions_data = event['questions']  # Pre-loaded questions
    responses_data = event['responses']  # Pre-collected responses
    templates_s3_key = event['templates_s3_key']

    if len(questions_data) > max_questions:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': f'Too many questions for Lambda. Maximum: {max_questions}'
            })
        }

    # Download templates from S3
    s3 = boto3.client('s3')
    s3_bucket = event['s3_bucket']

    templates_obj = s3.get_object(Bucket=s3_bucket, Key=templates_s3_key)
    templates_data = json.loads(templates_obj['Body'].read())

    # Load templates
    from karenina.answers.generator import load_answer_templates_from_json
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        json.dump(templates_data, tmp_file)
        templates_file = tmp_file.name

    try:
        templates = load_answer_templates_from_json(templates_file)

        # Run benchmark
        from karenina.benchmark.runner import run_benchmark
        results = run_benchmark(questions_data, responses_data, templates)

        # Serialize results
        serialized_results = {}
        for q_id, result in results.items():
            serialized_results[q_id] = result.model_dump()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Benchmark completed successfully',
                'results': serialized_results,
                'total_questions': len(results)
            })
        }

    finally:
        os.unlink(templates_file)

# Step Functions integration
def create_step_function_definition(s3_bucket: str) -> dict:
    """Create Step Functions definition for full pipeline."""

    return {
        "Comment": "Karenina Benchmarking Pipeline",
        "StartAt": "ExtractQuestions",
        "States": {
            "ExtractQuestions": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:region:account:function:karenina-extract-questions",
                "Parameters": {
                    "operation": "extract_questions",
                    "s3_bucket.$": "$.s3_bucket",
                    "s3_key.$": "$.s3_key",
                    "question_column.$": "$.question_column",
                    "answer_column.$": "$.answer_column",
                    "request_id.$": "$.request_id"
                },
                "Next": "GenerateTemplates"
            },
            "GenerateTemplates": {
                "Type": "Task",
                "Resource": "arn:aws:ecs:region:account:cluster/karenina-cluster",
                "Parameters": {
                    "TaskDefinition": "karenina-template-generation",
                    "LaunchType": "FARGATE",
                    "Overrides": {
                        "ContainerOverrides": [{
                            "Name": "karenina-container",
                            "Environment": [
                                {"Name": "S3_BUCKET", "Value.$": "$.s3_bucket"},
                                {"Name": "QUESTIONS_S3_KEY", "Value.$": "$.Payload.output_s3_key"},
                                {"Name": "REQUEST_ID", "Value.$": "$.request_id"}
                            ]
                        }]
                    }
                },
                "Next": "CollectResponses"
            },
            "CollectResponses": {
                "Type": "Task",
                "Resource": "arn:aws:ecs:region:account:cluster/karenina-cluster",
                "Parameters": {
                    "TaskDefinition": "karenina-response-collection",
                    "LaunchType": "FARGATE"
                },
                "Next": "RunBenchmark"
            },
            "RunBenchmark": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:region:account:function:karenina-benchmark",
                "Parameters": {
                    "operation": "run_benchmark",
                    "s3_bucket.$": "$.s3_bucket",
                    "templates_s3_key.$": "$.templates_s3_key",
                    "questions.$": "$.questions",
                    "responses.$": "$.responses"
                },
                "End": True
            }
        }
    }

# CloudFormation template for infrastructure
cloudformation_template = {
    "AWSTemplateFormatVersion": "2010-09-09",
    "Description": "Karenina Benchmarking Infrastructure",
    "Resources": {
        "KareninaBucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "BucketName": "karenina-benchmarking-data",
                "VersioningConfiguration": {
                    "Status": "Enabled"
                }
            }
        },
        "KareninaExecutionRole": {
            "Type": "AWS::IAM::Role",
            "Properties": {
                "AssumeRolePolicyDocument": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }]
                },
                "ManagedPolicyArns": [
                    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                ],
                "Policies": [{
                    "PolicyName": "KareninaS3Access",
                    "PolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [{
                            "Effect": "Allow",
                            "Action": ["s3:GetObject", "s3:PutObject"],
                            "Resource": {"Fn::Sub": "${KareninaBucket}/*"}
                        }]
                    }
                }]
            }
        },
        "KareninaLambdaFunction": {
            "Type": "AWS::Lambda::Function",
            "Properties": {
                "FunctionName": "karenina-benchmarking",
                "Runtime": "python3.9",
                "Handler": "lambda_function.lambda_handler",
                "Role": {"Fn::GetAtt": ["KareninaExecutionRole", "Arn"]},
                "Code": {
                    "ZipFile": "# Lambda function code here"
                },
                "Timeout": 300,
                "MemorySize": 1024,
                "Environment": {
                    "Variables": {
                        "S3_BUCKET": {"Ref": "KareninaBucket"}
                    }
                }
            }
        }
    }
}
```
