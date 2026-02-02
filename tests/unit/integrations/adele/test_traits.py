"""Unit tests for ADeLe trait conversion and API."""

import pytest

from karenina.integrations.adele import (
    ADELE_CODE_TO_NAME,
    ADELE_CODES,
    ADELE_NAME_TO_CODE,
    ADELE_TRAIT_NAMES,
    create_adele_rubric,
    get_adele_trait,
    get_adele_trait_by_code,
    get_all_adele_traits,
)
from karenina.schemas.domain.rubric import LLMRubricTrait, Rubric


class TestConstants:
    """Tests for ADeLe constants."""

    def test_adele_code_to_name_has_18_entries(self) -> None:
        """Verify all 18 ADeLe rubrics are mapped."""
        assert len(ADELE_CODE_TO_NAME) == 18

    def test_adele_trait_names_has_18_entries(self) -> None:
        """Verify all 18 trait names are available."""
        assert len(ADELE_TRAIT_NAMES) == 18

    def test_adele_codes_has_18_entries(self) -> None:
        """Verify all 18 codes are available."""
        assert len(ADELE_CODES) == 18

    def test_name_to_code_is_inverse(self) -> None:
        """Verify ADELE_NAME_TO_CODE is the inverse of ADELE_CODE_TO_NAME."""
        for code, name in ADELE_CODE_TO_NAME.items():
            assert ADELE_NAME_TO_CODE[name] == code

    def test_expected_codes_present(self) -> None:
        """Verify specific expected codes are present."""
        expected_codes = [
            "AS",
            "AT",
            "CEc",
            "CEe",
            "CL",
            "KNa",
            "KNc",
            "KNf",
            "KNn",
            "KNs",
            "MCr",
            "MCt",
            "MCu",
            "MS",
            "QLl",
            "QLq",
            "SNs",
            "VO",
        ]
        for code in expected_codes:
            assert code in ADELE_CODE_TO_NAME

    def test_expected_names_present(self) -> None:
        """Verify specific expected names are present."""
        expected_names = [
            "attention_and_scan",
            "atypicality",
            "mind_modelling",
            "volume",
        ]
        for name in expected_names:
            assert name in ADELE_TRAIT_NAMES


class TestGetAdeleTrait:
    """Tests for get_adele_trait function."""

    def test_get_attention_and_scan_trait(self) -> None:
        """Test getting attention_and_scan trait."""
        trait = get_adele_trait("attention_and_scan")

        assert isinstance(trait, LLMRubricTrait)
        assert trait.name == "attention_and_scan"
        assert trait.kind == "literal"
        assert trait.classes is not None
        assert len(trait.classes) == 6
        assert trait.higher_is_better is True

    def test_trait_has_correct_class_names(self) -> None:
        """Test that trait classes have correct names."""
        trait = get_adele_trait("attention_and_scan")

        expected_class_names = [
            "none",
            "very_low",
            "low",
            "intermediate",
            "high",
            "very_high",
        ]
        actual_class_names = list(trait.classes.keys())
        assert actual_class_names == expected_class_names

    def test_trait_classes_have_descriptions(self) -> None:
        """Test that each class has a non-empty description."""
        trait = get_adele_trait("attention_and_scan")

        for class_name, description in trait.classes.items():
            assert description, f"Class {class_name} has empty description"
            assert "Level" in description, f"Class {class_name} missing Level prefix"

    def test_trait_has_description_when_header_exists(self) -> None:
        """Test that trait description is set from header for rubrics with headers."""
        trait = get_adele_trait("attention_and_scan")

        # AS has a header
        assert trait.description is not None
        assert "attention" in trait.description.lower()

    def test_trait_has_no_description_when_no_header(self) -> None:
        """Test that trait description is None for rubrics without headers."""
        trait = get_adele_trait("atypicality")

        # AT has no header
        assert trait.description is None

    def test_min_max_score_derived_from_classes(self) -> None:
        """Test that min_score and max_score are correctly derived."""
        trait = get_adele_trait("attention_and_scan")

        assert trait.min_score == 0
        assert trait.max_score == 5  # 6 classes -> indices 0-5

    def test_get_trait_unknown_name_raises(self) -> None:
        """Test that unknown trait name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ADeLe trait name"):
            get_adele_trait("unknown_trait")

    def test_get_all_trait_names(self) -> None:
        """Test getting each trait by name."""
        for name in ADELE_TRAIT_NAMES:
            trait = get_adele_trait(name)
            assert trait.name == name
            assert trait.kind == "literal"


class TestGetAdeleTraitByCode:
    """Tests for get_adele_trait_by_code function."""

    def test_get_trait_by_code(self) -> None:
        """Test getting trait by original ADeLe code."""
        trait = get_adele_trait_by_code("AS")

        assert trait.name == "attention_and_scan"
        assert trait.kind == "literal"

    def test_get_all_traits_by_code(self) -> None:
        """Test getting each trait by code."""
        for code in ADELE_CODES:
            trait = get_adele_trait_by_code(code)
            expected_name = ADELE_CODE_TO_NAME[code]
            assert trait.name == expected_name

    def test_get_trait_unknown_code_raises(self) -> None:
        """Test that unknown code raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ADeLe code"):
            get_adele_trait_by_code("UNKNOWN")


class TestGetAllAdeleTraits:
    """Tests for get_all_adele_traits function."""

    def test_returns_18_traits(self) -> None:
        """Test that function returns all 18 traits."""
        traits = get_all_adele_traits()

        assert len(traits) == 18

    def test_all_traits_are_literal_kind(self) -> None:
        """Test that all traits have kind='literal'."""
        traits = get_all_adele_traits()

        for trait in traits:
            assert trait.kind == "literal"

    def test_all_traits_have_6_classes(self) -> None:
        """Test that all traits have exactly 6 classes."""
        traits = get_all_adele_traits()

        for trait in traits:
            assert trait.classes is not None
            assert len(trait.classes) == 6

    def test_all_traits_are_higher_is_better(self) -> None:
        """Test that all traits have higher_is_better=True."""
        traits = get_all_adele_traits()

        for trait in traits:
            assert trait.higher_is_better is True

    def test_trait_names_match_expected(self) -> None:
        """Test that returned traits have expected names."""
        traits = get_all_adele_traits()
        trait_names = {t.name for t in traits}

        assert trait_names == set(ADELE_TRAIT_NAMES)


class TestCreateAdeleRubric:
    """Tests for create_adele_rubric function."""

    def test_create_rubric_all_traits(self) -> None:
        """Test creating rubric with all traits."""
        rubric = create_adele_rubric()

        assert isinstance(rubric, Rubric)
        assert len(rubric.llm_traits) == 18
        assert len(rubric.regex_traits) == 0
        assert len(rubric.callable_traits) == 0
        assert len(rubric.metric_traits) == 0

    def test_create_rubric_selected_traits(self) -> None:
        """Test creating rubric with selected traits."""
        selected = ["attention_and_scan", "mind_modelling", "volume"]
        rubric = create_adele_rubric(trait_names=selected)

        assert len(rubric.llm_traits) == 3
        trait_names = {t.name for t in rubric.llm_traits}
        assert trait_names == set(selected)

    def test_create_rubric_single_trait(self) -> None:
        """Test creating rubric with single trait."""
        rubric = create_adele_rubric(trait_names=["atypicality"])

        assert len(rubric.llm_traits) == 1
        assert rubric.llm_traits[0].name == "atypicality"

    def test_create_rubric_unknown_trait_raises(self) -> None:
        """Test that unknown trait name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ADeLe trait name"):
            create_adele_rubric(trait_names=["unknown_trait"])

    def test_rubric_trait_names_method(self) -> None:
        """Test that rubric.get_trait_names() works correctly."""
        rubric = create_adele_rubric(trait_names=["attention_and_scan", "volume"])

        names = rubric.get_trait_names()
        assert set(names) == {"attention_and_scan", "volume"}


class TestTraitClassDescriptions:
    """Tests for class description formatting."""

    def test_class_descriptions_include_level_info(self) -> None:
        """Test that class descriptions include level number and label."""
        trait = get_adele_trait("attention_and_scan")

        for i, (_class_name, description) in enumerate(trait.classes.items()):
            assert f"Level {i}:" in description

    def test_class_descriptions_include_examples(self) -> None:
        """Test that class descriptions include examples."""
        trait = get_adele_trait("attention_and_scan")

        # At least some classes should have examples
        has_examples = False
        for description in trait.classes.values():
            if "Examples:" in description or "* " in description:
                has_examples = True
                break

        assert has_examples, "No class descriptions contain examples"

    def test_class_order_matches_level_order(self) -> None:
        """Test that class order in dict matches level order (0-5)."""
        trait = get_adele_trait("attention_and_scan")

        class_names = list(trait.classes.keys())
        expected_order = [
            "none",
            "very_low",
            "low",
            "intermediate",
            "high",
            "very_high",
        ]
        assert class_names == expected_order


class TestCaching:
    """Tests for caching behavior."""

    def test_same_trait_returns_equivalent_object(self) -> None:
        """Test that getting same trait twice returns equivalent objects."""
        trait1 = get_adele_trait("attention_and_scan")
        trait2 = get_adele_trait("attention_and_scan")

        # Should be equivalent (same content)
        assert trait1.name == trait2.name
        assert trait1.description == trait2.description
        assert trait1.classes == trait2.classes
