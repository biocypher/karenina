# Karenina Rubric Guide

Self-contained reference for proposing rubric traits after an error
analysis run. Bundled with every error-analysis directory so an
analyst agent can produce a well-formed proposal without prior
karenina experience.

## What a rubric is

A karenina rubric is a list of traits applied to every answer during
verification. Each trait produces a score or label for one narrow
qualitative property. Traits are independent: they do not compose,
they do not reference each other, and they do not share state.

## Scope and ground-truth context

This classifier is a diagnostic, not a correctness gate. It is
applied only to responses that verification has already flagged as
failures; the upstream pipeline has already decided the answer is
wrong. The classifier's job is to say *why* it failed, not *whether*
it failed.

In general, karenina rubric traits do not see the correct answer
(correctness is the job of answer templates). For this classifier
that constraint is relaxed: the question, the expected answer, the
requested unit, the tool schema, and any pipeline failure metadata
are all legitimate context when they are what distinguishes one
failure mode from another. `unit_mismatch`, for instance, is only
recognizable by comparing the response's unit to the expected one.

Keep the focus on *what the response did wrong* rather than
*whether the response is wrong*. A class like `incorrect_answer`
collapses every failure into one bucket and defeats the purpose of
the classifier.

## The proposed deliverable: a failure-mode classifier

The single rubric addition produced by this analysis is a
**failure-mode classifier**: one `LLMRubricTrait` of
`kind: "literal"` whose `classes` are the distinct failure modes
you identified. Applied downstream to any failing response, it
assigns the single class that best explains why that response
failed. The purpose is to let future runs aggregate failures by
cause rather than by the coarse technical category already in
INDEX.md, so that trends, regressions, and the impact of
benchmark changes become visible without re-reading every case.

Each class maps to one sub-pattern from REPORT.md.

Key properties:

  - The classifier is applied only to failing responses, downstream
    of whatever gating produces the failure set. Do not include a
    "no failure" or "clean" class.
  - Cap at around eight classes. Beyond that, the judge loses
    discrimination. When more sub-patterns surfaced, keep the most
    impactful by count and roll the rest into a single
    `other_pathology` class.
  - Class names must be mutually exclusive. A failing response
    should match exactly one class.

### How to write a class description

Each class description is read verbatim by the judge and is the
single artifact that determines whether the classifier works. Write
it as a short structured paragraph covering all four of the
following:

  1. **Pattern**: what the response does that puts it in this
     class. One or two sentences, framed as an observation the
     judge can check against the response text.
  2. **Signals**: concrete cues the judge can look for: specific
     phrasings, trace features, tool-call shapes, or artifacts.
     Name them, quote them where useful, list at least two.
  3. **Not this**: superficially similar patterns that belong to a
     different class, spelled out so the judge does not mis-route.
  4. **Disambiguation**: when this class is plausibly confusable
     with another class in the same classifier, state the concrete
     rule that decides which one applies. Skip this bullet only
     when no other class in the classifier is close.

The worked example below shows the expected depth. Thin
descriptions ("the model hallucinated") produce unreliable
classifications; write every class as if you were teaching a
fresh reader to recognize the pattern from scratch.

### LLMRubricTrait fields (literal kind)

  - `name` (str, required): unique identifier within the rubric.
    `failure_mode_classifier` is a fine default.
  - `description` (str, required): one-sentence framing of the
    classification task, read verbatim by the judge.
  - `kind`: set to `"literal"`.
  - `classes` (dict of str to str, required): mapping from class name
    to pattern description. Insertion order defines the index.
  - `higher_is_better` (bool, default `true`): kept for schema
    compatibility; directionality is rarely meaningful for a
    pathology classifier.

### Worked example

Class descriptions use YAML's literal block scalar (`|`) so newlines
are preserved and the four-part structure reads cleanly to the
judge.

```yaml
llm_traits:
  - name: failure_mode_classifier
    description: >
      The response below is known to be a failure of the benchmark
      question. Classify it into the single failure mode that best
      explains why it failed. Pick the class whose Pattern most
      closely matches what the response does; when two classes seem
      to fit, use their Disambiguation rules to decide. Assign
      exactly one class. If nothing fits, use `other_pathology`.
    kind: literal
    classes:
      hedges_then_refuses: |
        Pattern: the response opens with hedges or caveats about
        the model's capability, its access to information, or the
        inherent uncertainty of the question, and then either
        declines to give a direct answer or pivots into unrelated
        safety or capability boilerplate.
        Signals: phrases such as "I cannot be certain", "I do not
        have access to", "as an AI language model", "I would
        recommend consulting...", followed by no concrete answer.
        The answer field is either empty, a meta-statement about
        not answering, or a generic recommendation.
        Not this: a response that hedges briefly and then goes on
        to answer the question (hedging alone is not refusal).
        Not this either: refusing because of a safety-policy
        trigger with explicit policy language (classify as
        `other_pathology` and note it).
        Disambiguation: versus `echoes_question`, this class
        leaves the question substantively unanswered with caveats
        about the model; echoing instead restates the question as
        if that were the answer.
      echoes_question: |
        Pattern: the response repeats, paraphrases, or
        bullet-expands the question itself rather than supplying
        an answer. The answer field tracks the wording of the
        question rather than any external fact.
        Signals: openings such as "The question asks...",
        "You are asking whether...", or a near-verbatim restatement
        of the prompt; absence of new facts, numbers, or named
        entities that were not already in the question.
        Not this: a response that quotes the question briefly as
        scaffolding before answering it. Not this either: a
        response that restates the question to hedge about its
        answerability (that is `hedges_then_refuses`).
        Disambiguation: versus `hedges_then_refuses`, echoing
        contains no disclaimer about capability or access; the
        response simply never transitions from restatement to
        answer.
      ignored_tool_result: |
        Pattern: the model emits a tool call whose result is
        returned successfully, but the next assistant turn
        proceeds as though no result came back. Downstream
        reasoning never references the tool output and often
        re-asks the tool's own question in prose.
        Signals: a tool call followed by an assistant turn that
        says "Let me check", restates a plan identical to the
        pre-call state, or invents a value the tool already
        provided; absence of any quoted value, number, or name
        from the tool result in subsequent text.
        Not this: a tool call whose result was itself an error or
        malformed payload (those typically manifest as
        `malformed_tool_args` or as an `other_pathology` with a
        note on the error). Not this either: a result that was
        read correctly but interpreted wrongly (classify as
        `other_pathology`).
        Disambiguation: versus `malformed_tool_args`, the call
        here was well-formed and the result was valid; the
        failure is purely in the assistant's downstream handling.
      malformed_tool_args: |
        Pattern: the model issues a tool call whose argument
        object is syntactically invalid, of the wrong type for
        the tool's schema, or missing required keys. The tool
        either rejects the call or returns an error.
        Signals: JSON parse errors on tool input, schema
        validation errors, tool responses beginning with "Error:"
        or "ValidationError:", missing keys that are listed as
        required in the tool definitions under benchmark/; a
        single tool call with multiple subsequent retry calls
        exhibiting the same structural flaw.
        Not this: a call to a tool name that does not exist at
        all (classify as `other_pathology` and note). Not this
        either: a well-formed call whose result is later ignored
        (that is `ignored_tool_result`).
        Disambiguation: versus `ignored_tool_result`, the failure
        here occurs at invocation time, before any result could
        have been returned.
      recursion_limit_hit: |
        Pattern: the trace ends abruptly because the agent runtime
        reached its recursion or max-turns limit. The response is
        cut off mid-plan; no final assistant message is produced,
        or the final message is a framework error rather than
        model output.
        Signals: a `RecursionLimitError`, a `GraphRecursionError`,
        or an equivalent framework signal as the last trace entry;
        a loop of tool-call / tool-result pairs that repeats
        without forward progress; an assistant-message count at
        the cap configured by the run.
        Not this: a response that is merely long and rambling but
        terminates normally. Not this either: a model that stops
        early due to a stop-token (that is a content pathology).
        Disambiguation: this class is determined by the framework
        signal at the tail of the trace, not by the prose content
        of the assistant turns.
      boilerplate_placeholder: |
        Pattern: the parsed schema field is filled with a literal
        template placeholder string that the model copied from
        the answer template rather than with an actual answer.
        Signals: field values such as "<your answer here>",
        "TODO", "N/A", "Example value", or strings that
        recognizably match placeholder text visible in the files
        under benchmark/templates/; an answer that matches the
        template's example verbatim across multiple cases.
        Not this: a genuine but incorrect answer. Not this either:
        a hedging non-answer written in prose (that is
        `hedges_then_refuses`).
        Disambiguation: the placeholder is recognizable as a
        literal fragment of the template. When in doubt,
        cross-reference the raw template under benchmark/ to
        confirm the string appears there.
      unit_mismatch: |
        Pattern: the response provides the correct numeric
        magnitude but in the wrong unit (for example grams when
        the question asks for milligrams, Celsius instead of
        Kelvin, meters instead of feet).
        Signals: the numeric value matches the expected value
        scaled by a standard unit-conversion factor; the response
        includes a unit string that differs from the unit
        requested in the question text or template schema.
        Not this: a numerically wrong answer independent of unit
        (classify as `other_pathology`). Not this either: the
        correct answer expressed with a synonymous unit
        abbreviation (e.g. "mg" vs "milligrams").
        Disambiguation: this class requires a unit conversion to
        be identifiable. If no unit is stated by the response,
        default to `other_pathology`.
      other_pathology: |
        Pattern: the failure is genuine but does not fit any of
        the classes above.
        Signals: none specific. Use this class when no more
        precise class applies; it is the backstop, not the
        default.
        Not this: any failure that plausibly matches one of the
        specific classes above; prefer the specific class.
        Disambiguation: before assigning this class, verify that
        no other class fits even partially. When assigning it,
        include in your reasoning a one-line description of what
        the failure looked like; these notes drive future class
        proposals.
    higher_is_better: true
```

## Optional supplement: regex traits for cheap lexical patterns

When a sub-pattern is defined by a literal substring or a tight
regex (stop-token leakage, boilerplate markers, specific malformed
tool-call substrings), a `RegexRubricTrait` is essentially free to
evaluate and runs alongside the classifier. Propose one only when
the pattern is lexically precise; ambiguous content-level patterns
belong in the classifier's class descriptions.

### RegexRubricTrait fields

  - `name` (str, required).
  - `description` (str, recommended): what the pattern matches.
  - `pattern` (str, required): valid Python regex.
  - `case_sensitive` (bool, default `true`).
  - `invert_result` (bool, default `false`): set `true` when a match
    signals the pathological pattern, so that "no match = good".
  - `higher_is_better` (bool, default `true`).

### Worked example

```yaml
regex_traits:
  - name: stop_token_leakage
    description: >
      Matches when a special model stop-token appears inside the
      response body.
    pattern: "(<\\|endoftext\\|>|<stop>|<\\|eot_id\\|>)"
    case_sensitive: false
    invert_result: true
    higher_is_better: true
```

## Full YAML shape

A complete proposal combines the classifier with any regex traits:

```yaml
llm_traits:
  - name: failure_mode_classifier
    ...
regex_traits:
  - name: ...
    ...
```

Omit `regex_traits` entirely if no regex-precise pattern was found.

## Common pitfalls

  - **Clean / baseline class.** The classifier is applied only to
    failures; a "no failure" class is meaningless and confuses the
    judge.
  - **Too many classes.** Beyond around eight, accuracy drops. Roll
    low-count patterns into `other_pathology`.
  - **Class descriptions as labels, not instructions.** The judge
    reads each description verbatim. A one-line label ("hedging
    and refusal") teaches nothing; use the Pattern / Signals /
    Not this / Disambiguation structure from the worked example.
  - **Missing anti-patterns.** Without "Not this" and
    "Disambiguation" entries, the judge routes on surface
    similarity and misclassifies edge cases. Every class that has
    a plausible neighbour in the classifier needs both.
  - **Mutually ambiguous classes.** The classifier picks exactly one
    class. Two overlapping Patterns produce unstable labels;
    sharpen them or merge the classes.
  - **Overly narrow regex.** A regex that matched only one failing
    case will not generalize. Require at least three matching cases
    before proposing it.
  - **Using ground truth as a correctness check.** The classifier
    runs only after verification has already decided the response
    is wrong. Descriptions like "the answer is incorrect" or "the
    response names the wrong protein" make the classifier a
    redundant correctness gate and collapse many distinct failure
    modes into one. Use ground-truth context only to distinguish
    one failure mode from another (for example, unit mismatch vs
    numerical error), not to re-prove the response is wrong.
