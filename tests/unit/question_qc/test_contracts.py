"""Unit tests for QC JSON contracts."""

from karenina.question_qc.contracts import parse_proposal, parse_review, parse_validation
from karenina.question_qc.models import QcClassification


def test_parse_proposal_minimal() -> None:
    text = '{"decision":"propose","witness":"MATCH (n) RETURN n","explanation":"ok"}'
    p = parse_proposal(text)
    assert p.decision == "propose"
    assert p.witness == "MATCH (n) RETURN n"
    assert p.explanation == "ok"


def test_parse_proposal_query_alias() -> None:
    text = '{"decision":"propose","query":"SELECT 1","explanation":"x"}'
    p = parse_proposal(text)
    assert p.witness == "SELECT 1"


def test_parse_proposal_abandon() -> None:
    text = '{"decision":"give_up","witness":"","give_up_reason":"no path","explanation":""}'
    p = parse_proposal(text)
    assert p.abandoned
    assert p.abandon_reason == "no path"


def test_parse_proposal_markdown_fence() -> None:
    text = '```json\n{"decision":"propose","witness":"W","explanation":"e"}\n```'
    p = parse_proposal(text)
    assert p.witness == "W"


def test_parse_validation_gate() -> None:
    ok = parse_validation('{"supports_claim":true,"reasoning":"r","quality_issues":""}')
    assert ok.passes_evidence_gate
    bad = parse_validation('{"supports_claim":true,"reasoning":"r","quality_issues":"too broad"}')
    assert not bad.passes_evidence_gate
    no = parse_validation('{"supports_claim":false,"reasoning":"r","quality_issues":""}')
    assert not no.passes_evidence_gate


def test_parse_validation_answers_question_alias() -> None:
    v = parse_validation('{"answers_question":true,"reasoning":"r","quality_issues":""}')
    assert v.supports_claim is True


def test_parse_review_aliases() -> None:
    r = parse_review('{"classification":"pass","reasoning":"good","quality_issues":""}')
    assert r.classification == QcClassification.SUPPORTED
    r2 = parse_review('{"classification":"malformed","reasoning":"bad","quality_issues":""}')
    assert r2.classification == QcClassification.ILL_FORMED
