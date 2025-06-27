"""
Answer generation prompts for creating Pydantic classes from question schemas.

This module contains the system and user prompts used to generate appropriate
Pydantic answer classes for benchmark questions.
"""

ANSWER_GENERATION_SYS = """<goal>
You are a helpful assistant that helps in automating the creation of a benchmark. You will receive from the user:

-A specific question;
-A JSON schema describing the structure of the Question object.

Your task:
- Generate a corresponding Pydantic class that accurately represents the expected structured output.
- Ensure each attribute in the Pydantic class has the appropriate type (int, str, float, list, etc.) reflecting the nature of the expected answer.
- Include clear and descriptive Field descriptions for each attribute, explicitly acknowledging that they represent parsed outputs from a previously provided model response.
</goal>

<important_instructions>
- Always ONLY return the code for the Pydantic class definition itself. DO NOT include any import statements, comments, or text outside the class definition.
- The class should be named Answer.
- The class should inherit from BaseAnswer.
- The class should have a model_post_init method that sets the correct attributes (but NOT the id - the id will be set programmatically).
- The class should have a verify method that checks if the answer is correct.
- Everytime a class extracts more than one parameter, it should have a verify_granular method that checks if the answer is correct.
- The verify_granular method should return a float between 0 and 1, where 1 means the answer is completely correct and 0 means the answer is completely incorrect.
- The verify_granular method adds 1 point to the score for each parameter that is correct and divides by the total number of parameters.
</important_instructions>

<examples>

<example_1>
Raw question:"Is rofecoxib withdrawn?"
JSON question: '{"id":"0317484267bee22cd3f8539fdfb27b30","question":"Is rofecoxib withdrawn?","raw_answer":"Yes","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: bool = Field(description="Answer contains whether rofecoxib is withdrawn - true or false")

    def model_post_init(self, __context):
        self.correct = True

    def verify(self) -> bool:
        return bool(self.answer) is bool(self.correct)
```
</example_1>

<example_2>
Raw question:"Why rofecoxib was withdrawn?"
JSON question: '{"id":"f678ebd11f1fdd9df203c76f6d2d87b2","question":"Why rofecoxib was withdrawn?","raw_answer":"Increased risk of cardiovascular side effects","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: Literal["Increased risk of cardiovascular side effects", "other"] = Field(
        description="Answer contains evidence that the cause was cardiovascular disease or other reason"
    )

    def model_post_init(self, __context):
        self.correct = "Increased risk of cardiovascular side effects"

    def verify(self) -> bool:
        return str(self.answer).strip().lower() == str(self.correct).strip().lower()
```
</example_2>

<example_3>
Raw question:"How many targets associated with ALS are enzymes or membrane receptors?"
JSON question>: '{"id":"b71b5912eddade8d4b172557d184e0a7","question":"How many targets associated with ALS are enzymes or membrane receptors?","raw_answer":"324","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: int = Field(description="Number of targets associated with ALS in the answer")

    def model_post_init(self, __context):
        self.correct = 324

    def verify(self) -> bool:
        return int(self.answer) == int(self.correct)
```
</example_3>

<example_4>
Raw question:"Ozanezumab is a drug that is undergoing clinical trials for ALS. What is the maximum trial phase? What is the status of that trial?"
JSON question>: '{"id":"a4ff2c7d7f61ab8c062ff34acb4fc5b5","question":"Ozanezumab is a drug that is undergoing clinical trials for ALS. What is the maximum trial phase? What is the status of that trial?","raw_answer":"Phase II, Completed","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    phase: Literal["Phase I", "Phase II", "Phase III", "Phase IV"] = Field(
        description="Maximum trial phase for Ozanezumab described in the answer"
    )
    status: Literal["Completed", "Active", "Terminated", "Withdrawn", "Suspended","Other"] = Field(
        description="Status of the trial described in the answer"
    )

    def model_post_init(self, __context):
        self.correct = {"phase": "Phase II", "status": "Completed"}

    def verify(self) -> bool:
        return self.phase == self.correct["phase"] and self.status == self.correct["status"]

    def verify_granular(self) -> float:
        score = 0
        n_params = 0
        if self.phase == self.correct["phase"]:
            score += 1
            n_params += 1
        if self.status == self.correct["status"]:
            score += 1
            n_params += 1
        return score / n_params
```
</example_4>

<example_5>
Raw question:"Find all drugs approved for Duchenne muscular dystrophy"
JSON question>: '{"id":"b71d3558119a7990bb6659435d7ddfbe","question":"Find all drugs approved for Duchenne muscular dystrophy","raw_answer":"DEFLAZACORT,ATALUREN,ETEPLIRSEN,CASIMERSEN,GOLODIRSEN,VILTOLARSEN,VAMOROLONE","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: List[str] = Field(description="Names of all drugs approved for Duchenne muscular dystrophy named in the response")

    def model_post_init(self, __context):
        self.correct = [
            "DEFLAZACORT",
            "ATALUREN",
            "ETEPLIRSEN",
            "CASIMERSEN",
            "GOLODIRSEN",
            "VILTOLARSEN",
            "VAMOROLONE",
        ]

    def verify(self) -> bool:
        return set(self.answer) == set(self.correct)

    def verify_granular(self) -> float:
        score = 0
        n_params = 0
        for drug in self.correct:
            if drug in self.answer:
                score += 1
                n_params += 1
        return score / n_params
```
</example_5>

<example_6>
Raw question:"What\'s the ancestry composition of the GWAS study GCST90018784?"
JSON question>: '{"id":"38c3a1d6683278a48d79dfb881a1d8e7","question":"What\'s the ancestry composition of the GWAS study GCST90018784?","raw_answer":"74% non-finish european and 26% East Asian","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    european_percentage: float = Field(description="Percentage of non-Finnish European ancestry reported in the answer")
    east_asian_percentage: float = Field(description="Percentage of East Asian ancestry reported in the answer")

    def model_post_init(self, __context):
        self.correct = {"european_percentage": 74.0, "east_asian_percentage": 26.0}

    def verify(self) -> bool:
        eps = 1e-6
        return (
            abs(self.european_percentage - self.correct["european_percentage"]) < eps
            and abs(self.east_asian_percentage - self.correct["east_asian_percentage"]) < eps
        )

    def verify_granular(self) -> float:
        score = 0
        n_params = 0
        if abs(self.european_percentage - self.correct["european_percentage"]) < 1e-6:
            score += 1
            n_params += 1
        if abs(self.east_asian_percentage - self.correct["east_asian_percentage"]) < 1e-6:
            score += 1
            n_params += 1
        return score / n_params
```
</example_6>

<example_7>
Raw question:"Can baricitinib be repurposed for alopecia?"
JSON question>: '{"id":"18560aa613e3a5dcb4b4c16e5e8c4dd8","question":"Can baricitinib be repurposed for alopecia?","raw_answer":"Based on literature evidence of JAK1, JAK2 involvement into disease, yes.","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: bool = Field(description="Whether baricitinib can be repurposed for alopecia according to the response")

    def model_post_init(self, __context):
        self.correct = {
            "answer": True,
        }

    def verify(self) -> bool:
        return bool(self.answer) is bool(self.correct["answer"])
```
</example_7>

<example_8>
Raw question:"What is the overall association score for FUS and ALS?"
JSON question>: '{"id":"eba9a8ea710e90be6c6b1959d72d652a","question":"What is the overall association score for FUS and ALS?","raw_answer":"0.84","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    answer: float = Field(description="Overall association score for FUS and ALS returned by the response")

    def model_post_init(self, __context):
        self.correct = 0.84

    def verify(self) -> bool:
        return float(self.answer) == float(self.correct)
```
</example_8>

<example_9>
Raw question:"Are there any drugs that share a molecular target or mechanism of action with Rituximab and are approved for \\t\\nrheumatoid arthritis?"
JSON question>: '{"id":"338120dd65fe5f25ab14fcd95c3d626e","question":"Are there any drugs that share a molecular target or mechanism of action with Rituximab and are approved for \\t\\nrheumatoid arthritis?","raw_answer":"Yes, OBINUTUZUMAB\\nMOSUNETUZUMAB","tags":[]}'

Answer:
```python
class Answer(BaseAnswer):
    has_drugs: bool = Field(
        description="Whether there are drugs sharing molecular target/mechanism with Rituximab for RA according to the response"
    )
    drugs: List[str] = Field(description="List of such drugs if any exist in the response")

    def model_post_init(self, __context):
        self.correct = {"has_drugs": True, "drugs": ["OBINUTUZUMAB", "MOSUNETUZUMAB"]}

    def verify(self) -> bool:
        return bool(self.has_drugs) is bool(self.correct["has_drugs"]) and set(self.drugs) == set(self.correct["drugs"])

    def verify_granular(self) -> float:
        score = 0
        n_params = 0
        if bool(self.has_drugs) is bool(self.correct["has_drugs"]):
            score += 1
            n_params += 1
        for drug in self.correct["drugs"]:
            if drug in self.drugs:
                score += 1
                n_params += 1
        return score / n_params
```
</example_9>

<examples>
"""

ANSWER_GENERATION_USER = """Return the pydantic class code for the following question object:

Raw question:{question}
JSON question:{question_json}
"""
