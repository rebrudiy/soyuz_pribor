import asyncio
from collections.abc import Mapping
from copy import deepcopy

import pytest

from application.check_factuality import check_factuality
from application.llm import LLMError


def _draft() -> dict[str, object]:
    return {
        "seo_title": "Контрольный прибор",
        "seo_description": "Измеряет давление.",
        "search_queries": ["Контрольный прибор", "прибор давления"],
        "commercial_block": "Для измерения давления.",
        "clarifications": ["Уточните питание."],
    }


class FakeReviewer:
    """Test the review contract, not a real model's factual accuracy."""

    def __init__(self, response: dict[str, object] | LLMError) -> None:
        self.response = response
        self.calls: list[tuple[dict[str, object], str, dict[str, object]]] = []

    async def generate(
        self,
        data: Mapping[str, object],
        *,
        instructions: str,
        schema: Mapping[str, object],
    ) -> dict[str, object]:
        self.calls.append((deepcopy(dict(data)), instructions, deepcopy(dict(schema))))
        if isinstance(self.response, LLMError):
            raise self.response
        return deepcopy(self.response)


def test_sends_original_source_and_all_draft_fields_for_one_review() -> None:
    source = {"name": "Контрольный прибор", "specs": {"offset": 0, "enabled": False}}
    draft = _draft()
    original_source, original_draft = deepcopy(source), deepcopy(draft)
    client = FakeReviewer({"issues": []})

    result = asyncio.run(check_factuality(source, draft, client=client))

    assert result == {"status": "no_unsupported_claims_found", "issues": []}
    assert source == original_source
    assert draft == original_draft
    assert len(client.calls) == 1
    data = client.calls[0][0]
    assert data == {"source": original_source, "draft": original_draft}


def test_empty_source_and_honest_missing_facts_can_receive_no_issues() -> None:
    draft = {
        "seo_title": "Данных нет",
        "seo_description": "Сведения отсутствуют.",
        "search_queries": [],
        "commercial_block": "Назначение неизвестно.",
        "clarifications": ["Уточните характеристики."],
    }
    client = FakeReviewer({"issues": []})

    result = asyncio.run(check_factuality({}, draft, client=client))

    assert result["status"] == "no_unsupported_claims_found"
    assert client.calls[0][0] == {"source": {}, "draft": draft}


@pytest.mark.parametrize("field", list(_draft()))
def test_exact_claim_spans_are_accepted_in_every_text_field(field: str) -> None:
    draft = _draft()
    value = draft[field]
    claim = value[0] if isinstance(value, list) else value
    issue = {"field": field, "claim": claim, "reason": "Проверка указанного фрагмента."}
    client = FakeReviewer({"issues": [issue]})

    result = asyncio.run(check_factuality({}, draft, client=client))

    assert result == {"status": "needs_review", "issues": [issue]}


def _issue() -> dict[str, object]:
    return {
        "field": "seo_description",
        "claim": "Измеряет давление.",
        "reason": "Назначение не подтверждено исходными данными.",
    }


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"issues": None},
        {"issues": [], "status": "supported"},
        {"issues": [None]},
        {"issues": [_issue() | {"field": "unknown"}]},
        {"issues": [_issue() | {"claim": " \t"}]},
        {"issues": [_issue() | {"reason": " \n"}]},
        {"issues": [_issue() | {"claim": "Текст, которого нет в черновике."}]},
        {
            "issues": [
                _issue()
                | {
                    "field": "search_queries",
                    "claim": "Контрольный прибор прибор давления",
                }
            ]
        },
    ],
)
def test_malformed_review_is_an_error_instead_of_an_empty_pass(
    report: dict[str, object],
) -> None:
    with pytest.raises(LLMError):
        asyncio.run(check_factuality({}, _draft(), client=FakeReviewer(report)))


def test_model_error_propagates_without_retry() -> None:
    error = LLMError("Review request failed.")
    client = FakeReviewer(error)

    with pytest.raises(LLMError) as caught:
        asyncio.run(check_factuality({}, _draft(), client=client))

    assert caught.value is error
    assert len(client.calls) == 1
