"""Prompt template for standalone Claude Code generation."""

GENERATION_PROMPT_TEMPLATE = """You are generating adversarial answers for an Open Targets Platform benchmark question.

QUESTION DETAILS
================
ID:           {item_id}
Area:         {area}
Type:         {question_type}
Question:     {question}
Ground Truth: {ground_truth}

TASK
====
Produce TWO adversarial wrong answers:

1. HARD adversarial: A wrong answer that looks highly plausible. It MUST be a real
   value that exists on the Open Targets Platform, in the same narrow domain as the
   correct answer. An expert would need to verify carefully to distinguish it from
   the truth. For example: a different drug targeting the same pathway, a nearby
   variant in the same gene region, a related disease with similar phenotype.

2. EASY adversarial: A wrong answer still in the biomedical domain but clearly
   incorrect to someone with basic knowledge. It should be a real entity/value
   but from a noticeably different area. For example: a drug for a completely
   unrelated disease, a gene from a different biological system, a variant on
   a different chromosome.

RESEARCH METHOD
===============
You have access to the Open Targets MCP tools. They are deferred tools, so you
MUST load them first by calling:

  ToolSearch(query="select:mcp__open-targets__search_entities,mcp__open-targets__get_open_targets_graphql_schema,mcp__open-targets__query_open_targets_graphql", max_results=5)

After loading, follow this workflow:

Step 1: Search for key entities in the question.
   Tool: mcp__open-targets__search_entities
   Search for genes, diseases, drugs, or variants mentioned in the question.

Step 2: Get the relevant schema section.
   Tool: mcp__open-targets__get_open_targets_graphql_schema
   Use the category most relevant to the question type (e.g. "drug-mechanisms",
   "genetic-associations", "disease-phenotypes", "variant-annotation", etc.)

Step 3: Query the platform for neighborhood data.
   Tool: mcp__open-targets__query_open_targets_graphql
   Explore data around the correct answer:
   - Other entries in the same field/table as the ground truth
   - Related entities that share properties with the correct answer
   - Entities that a non-expert might confuse with the correct answer

Use the data you retrieve to select adversarial answers that are grounded in
real platform content, not invented.

RETRY POLICY: If an MCP tool call fails (timeout, server error), wait 15 seconds
and retry up to 3 times. If all retries fail, fall back to web search as a last
resort. If web search also fails, use your domain knowledge but note this clearly
in the MCP DATA USED section.

OUTPUT
======
Write the result to the file at this exact path using the Write tool:
{output_path}

The file MUST use EXACTLY this structure (including the --- delimiters):

--- ADVERSARIAL SAMPLE ---
ID: {item_id}
Area: {area}
Type: {question_type}
Question: {question}
Ground Truth: {ground_truth}

--- HARD ADVERSARIAL ---
Answer: [your hard adversarial answer here]
Reasoning: [2-4 sentences. What MCP data led you to this choice. Why it is plausible. What distinguishes it from the ground truth.]

--- EASY ADVERSARIAL ---
Answer: [your easy adversarial answer here]
Reasoning: [2-4 sentences. Why this is wrong. Why it would be easy for a domain-aware person to reject. What makes it still in-domain enough to be a valid distractor.]

--- MCP DATA USED ---
[Summarize: which entities you searched, which queries you ran, what key data points informed your choices. If you fell back to web search or knowledge, state that here.]

Do NOT output any text to the conversation. Just write the file and confirm it was written.
"""

__all__ = ["GENERATION_PROMPT_TEMPLATE"]
