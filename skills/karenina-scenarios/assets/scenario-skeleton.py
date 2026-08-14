"""Two-node aspirin knowledge scenario with branching.

Demonstrates the core scenario builder API: nodes, conditional edges,
outcome criteria (sugar functions plus a manual TurnCheck with a
turn_at scope selector), and validation.
"""

from karenina.scenario import Scenario, END
from karenina.scenario.sugar import all_of, last_turn
from karenina.schemas.scenario.checks import TurnCheck
from karenina.schemas.primitives.scope import TurnAt
from karenina.schemas.primitives import BooleanMatch, ContainsAll, ExactMatch
from karenina.schemas.entities.question import Question

# --- Questions ---

q_mechanism = Question(
    question="What is the mechanism of action of aspirin?",
    raw_answer="Irreversible inhibition of cyclooxygenase (COX) enzymes",
    answer_template="""
class AspirinMOA(BaseAnswer):
    mechanism: str = VerifiedField(
        description="The mechanism of action of aspirin",
        ground_truth="Irreversible inhibition of cyclooxygenase (COX) enzymes",
        verify_with=ContainsAll(substrings=["cyclooxygenase"], normalize=["lowercase"]),
    )
""",
)

q_followup = Question(
    question="Which COX isoform is primarily responsible for platelet aggregation?",
    raw_answer="COX-1",
    answer_template="""
class COXIsoform(BaseAnswer):
    isoform: str = VerifiedField(
        description="The COX isoform responsible for platelet aggregation",
        ground_truth="COX-1",
        verify_with=ExactMatch(),
    )
""",
)

# --- Build the scenario graph ---

s = Scenario("aspirin-knowledge", description="Two-step aspirin knowledge probe")

# Add nodes
s.add_node("ask_moa", question=q_mechanism)
s.add_node("ask_isoform", question=q_followup)

# Add edges: correct answer leads to followup; wrong answer ends
s.add_edge("ask_moa", "ask_isoform", when={"verify_result": True})
s.add_edge("ask_moa", END)  # Unconditional fallback

# Followup always terminates
s.add_edge("ask_isoform", END)

# Set entry point
s.set_entry("ask_moa")

# --- Outcome criteria ---

# turn_at() is a scope SELECTOR, not a check: use it inside TurnCheck(scope=...).
first_turn_ok = TurnCheck(
    scope=TurnAt(index=0),
    field="verify_result",
    expected=True,
    verify_with=BooleanMatch(),
)

s.add_outcome(
    "mechanism_correct",
    first_turn_ok,
    description="First turn correctly identifies COX inhibition",
)

s.add_outcome(
    "full_knowledge",
    all_of(
        last_turn(verify_result=True),
        TurnCheck(
            scope=TurnAt(index=0),
            field="verify_result",
            expected=True,
            verify_with=BooleanMatch(),
        ),
    ),
    description="Both mechanism and isoform questions answered correctly",
)

# --- Validate ---

defn = s.validate()
print(f"Scenario '{defn.name}' validated with {len(defn.nodes)} nodes and {len(defn.edges)} edges")
