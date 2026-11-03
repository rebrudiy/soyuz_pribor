import asyncio
from collections.abc import Mapping
from copy import deepcopy

import pytest

from application.generate_product import generate_product
from application.llm import LLMError, LLMNotConfigured
from domain.product import InvalidProduct


def _draft() -> dict[str, object]:
    return {
        "seo_title": "Манометр DM02",
        "seo_description": "Описание. " * 100,
        "search_queries": ["манометр DM02", "манометр давления", "Манотомь DM02"],
        "commercial_block": "Для специалистов, которым нужен контроль давления.",
        "clarifications": ["Уточните питание."],
    }


class FakeClient:
    def __init__(
        self,
        result: dict[str, object] | LLMError,
        *,
        review: dict[str, object] | LLMError | None = None,
    ) -> None:
        self.result = result
        self.review = {"issues": []} if review is None else review
        self.calls: list[tuple[dict[str, object], str, dict[str, object]]] = []

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        self.calls.append((deepcopy(dict(data)), instructions, deepcopy(dict(schema))))
        result = self.review if "issues" in schema["properties"] else self.result
        if isinstance(result, LLMError):
            raise result
        return deepcopy(result)


def test_generation_preserves_input_and_builds_characteristics_from_supplied_specs() -> (
    None
):
    data = {
        "name": " Манометр DM02 ",
        "manufacturer": "Манотомь",
        "specs": {
            "range": "0-1,6 МПа",
            "power": None,
            "note": " \t",
            "offset": 0,
            "enabled": False,
        },
        "description": "Измерение давления.",
        "source_note": {"source": "supplied passport"},
    }
    original = deepcopy(data)
    client = FakeClient(_draft())

    result = asyncio.run(generate_product(data, client=client))

    assert data == original
    assert result["product"] == original
    assert set(result["missing_fields"]) == {"specs.power", "specs.note"}
    assert result["draft"] == _draft() | {
        "key_characteristics": [
            {"name": "range", "value": "0-1,6 МПа"},
            {"name": "power", "value": "данных нет"},
            {"name": "note", "value": "данных нет"},
            {"name": "offset", "value": 0},
            {"name": "enabled", "value": False},
        ]
    }
    assert result["warnings"] == []
    assert len(client.calls) == 2
    sent_data, instructions, schema = client.calls[0]
    assert sent_data == original
    assert instructions.strip()
    assert "key_characteristics" not in schema["properties"]


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"name": "", "manufacturer": None, "specs": None, "description": " \t"},
    ],
)
def test_incomplete_data_still_reaches_the_model(data: dict[str, object]) -> None:
    client = FakeClient(_draft())

    result = asyncio.run(generate_product(data, client=client))

    assert result["product"] == data
    assert set(result["missing_fields"]) == {
        "name",
        "manufacturer",
        "specs",
        "description",
    }
    assert result["draft"]["key_characteristics"] == []
    assert result["draft"]["clarifications"] == _draft()["clarifications"]
    assert len(client.calls) == 2
    assert client.calls[0][0] == data


@pytest.mark.parametrize(
    ("changes", "warning_count"),
    [
        (
            {
                "seo_title": "x" * 71,
                "seo_description": "x" * 799,
                "search_queries": ["one", "two"],
            },
            3,
        ),
        (
            {"seo_description": "x" * 1201, "search_queries": ["query"] * 6},
            2,
        ),
        (
            {"seo_title": "x" * 70, "seo_description": "x" * 800},
            0,
        ),
        (
            {"seo_description": "x" * 1200, "search_queries": ["query"] * 5},
            0,
        ),
    ],
)
def test_quality_limits_add_warnings_without_rewriting_or_rejecting_the_draft(
    changes: dict[str, object], warning_count: int
) -> None:
    draft = _draft() | changes
    client = FakeClient(draft)

    result = asyncio.run(generate_product({}, client=client))

    assert result["product"] == {}
    assert result["draft"] == draft | {"key_characteristics": []}
    assert len(result["warnings"]) == warning_count
    assert all(
        isinstance(warning, str) and warning.strip() for warning in result["warnings"]
    )
    assert len(client.calls) == 2


def test_factuality_review_compares_original_input_with_generated_draft() -> None:
    data = {"name": "DM02", "specs": {"range": "0-1,6 МПа"}}
    draft = _draft() | {"commercial_block": "Гарантия 24 месяца."}
    issue = {
        "field": "commercial_block",
        "claim": "Гарантия 24 месяца.",
        "reason": "The source does not specify a warranty.",
    }
    client = FakeClient(draft, review={"issues": [issue]})

    result = asyncio.run(generate_product(data, client=client))

    assert client.calls[1][0] == {"source": data, "draft": draft}
    assert result["factuality"] == {"status": "needs_review", "issues": [issue]}
    assert result["draft"]["commercial_block"] == draft["commercial_block"]


def test_failed_review_does_not_return_an_unchecked_success() -> None:
    error = LLMError("Reviewer unavailable")
    client = FakeClient(_draft(), review=error)

    with pytest.raises(LLMError) as caught:
        asyncio.run(generate_product({}, client=client))

    assert caught.value is error
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("changes", "expected_warning"),
    [
        ({"seo_title": " \t"}, "seo_title is blank."),
        ({"seo_description": " " * 800}, "seo_description is blank."),
        ({"commercial_block": "\n"}, "commercial_block is blank."),
        (
            {"search_queries": ["one", " ", "three"]},
            "Search queries contain blank entries.",
        ),
    ],
)
def test_blank_content_is_reported_without_rejecting_the_draft(
    changes: dict[str, object], expected_warning: str
) -> None:
    draft = _draft() | changes

    result = asyncio.run(generate_product({}, client=FakeClient(draft)))

    assert result["warnings"] == [expected_warning]
    assert result["draft"] == draft | {"key_characteristics": []}


@pytest.mark.parametrize(
    "draft",
    [
        pytest.param(
            {
                key: value
                for key, value in _draft().items()
                if key != "commercial_block"
            },
            id="missing-field",
        ),
        pytest.param(_draft() | {"seo_title": 42}, id="wrong-text-type"),
        pytest.param(
            _draft() | {"search_queries": ["valid", 42]}, id="wrong-query-item-type"
        ),
        pytest.param(
            _draft() | {"clarifications": "not a list"}, id="wrong-clarifications-type"
        ),
        pytest.param(
            _draft()
            | {"key_characteristics": [{"name": "invented range", "value": "999 MPa"}]},
            id="unexpected-model-field",
        ),
    ],
)
def test_rejects_malformed_model_output(draft: dict[str, object]) -> None:
    with pytest.raises(LLMError):
        asyncio.run(generate_product({}, client=FakeClient(draft)))


def test_missing_client_is_reported_as_a_configuration_error() -> None:
    with pytest.raises(LLMNotConfigured):
        asyncio.run(generate_product({}, client=None))


def test_model_errors_are_propagated() -> None:
    error = LLMError("The model request failed.")
    client = FakeClient(error)

    with pytest.raises(LLMError) as caught:
        asyncio.run(generate_product({}, client=client))

    assert caught.value is error
    assert len(client.calls) == 1


def test_invalid_input_is_rejected_before_using_the_client() -> None:
    client = FakeClient(_draft())
    invalid = {"manufacturer": 42}

    with pytest.raises(InvalidProduct):
        asyncio.run(generate_product(invalid, client=client))
    with pytest.raises(InvalidProduct):
        asyncio.run(generate_product(invalid, client=None))

    assert client.calls == []
