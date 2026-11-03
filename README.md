# Союз Прибор

Прототип на Python: принимает JSON с данными о приборе и создаёт черновик
B2B-карточки на русском. Второй запрос к модели проверяет утверждения по исходному JSON.
Технические характеристики копируются из входных данных. Пропуски не блокируют генерацию.

## Запуск

Нужен Python 3.12+.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade "pip>=25.1"
.venv/bin/python -m pip install --group dev
```

Через подписку ChatGPT и установленный [Codex CLI](https://learn.chatgpt.com/docs/non-interactive-mode):

```bash
codex login
.venv/bin/python main.py --codex
```

Или через [ключ OpenAI API](https://platform.openai.com/api-keys): скопируйте
`.env.example` в `.env`, укажите `OPENAI_API_KEY` и `OPENAI_MODEL`, затем запустите:

```bash
.venv/bin/python main.py
```

Зависимости описаны в `pyproject.toml`:
`app` — запуск, `dev` — запуск и разработка.

API: http://127.0.0.1:8000/docs. Пример запроса:

В терминале сервера видны этапы и время запроса. Curl получает JSON после генерации и проверки фактов.

```bash
curl --fail-with-body http://127.0.0.1:8000/products \
  -H 'Content-Type: application/json' \
  --data-binary @examples/assignment-sample.json
```

Реальные ответы через Codex CLI (`gpt-6.1-sol`, 03.11.2026):
[DM 02](examples/assignment-sample-result.json),
[WIKA](examples/wika-9022597-result.json).
[Проверка выдуманной гарантии](examples/factuality-probe.json) содержит изменённый
черновик и реальное замечание модели.

## Вход и ответ

`POST /products` принимает JSON-объект, включая `{}`. Поля `name`, `manufacturer`,
`description` — строки или `null`; `specs` — объект или `null`. Все поля необязательны.
Дополнительные поля сохраняются. Значения, требующие высокой точности, передавайте строками.

| Поле ответа | Содержимое |
| --- | --- |
| `product` | Исходные данные. |
| `draft` | SEO-заголовок и описание, поисковые запросы, коммерческий блок, уточнения и характеристики. |
| `missing_fields` | Недостающие данные; ноль и `false` не считаются пропусками. |
| `warnings` | Пустые тексты, повторы запросов, нарушения длины или количества. |
| `factuality` | Статус проверки и спорные утверждения с причинами. |

Цели: заголовок до 70 символов, описание 800–1200 символов, 3–5 запросов.
Нарушения дают предупреждения. Неизвестные характеристики обозначаются как `данных нет`.
`needs_review` означает наличие замечаний; `no_unsupported_claims_found` — их отсутствие.
Черновик возвращается без автоматического исправления. Оба статуса требуют проверки человеком.

Ошибки: `415` — тип содержимого; `422` — неверный JSON, типы полей, чрезмерная
вложенность или потеря точности чисел; `500` — внутренняя ошибка;
`502` — сбой генерации или проверки; `503` — нет настроек API.

## Структура

- [domain/product.py](domain/product.py) — данные товара и пропуски.
- [application/generate_product.py](application/generate_product.py) — сценарий генерации.
- [application/product_prompt.py](application/product_prompt.py) — инструкции и схема ответа.
- [application/check_factuality.py](application/check_factuality.py) — проверка утверждений.
- [application/llm.py](application/llm.py) — протокол клиента.
- [infrastructure/openai_client.py](infrastructure/openai_client.py) — OpenAI SDK.
- [presentation/http.py](presentation/http.py) — HTTP; [main.py](main.py) — сборка приложения.

Домен и приложение не зависят от FastAPI и OpenAI SDK — зависимости направлены внутрь.

## Проверки

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
```

[CI](.github/workflows/ci.yml) выполняет эти проверки без ключа API и обращений к модели.
