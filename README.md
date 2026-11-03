# Soyuz Pribor

Local product-card draft generator. Uses Python 3.12+ and pip 25.1+.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --group dev
```

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-5.4-mini
```

`.env` is ignored by Git. Do not commit your key.

```bash
.venv/bin/python main.py
```

Open http://127.0.0.1:8000/docs and send a JSON object to `POST /products`.
All product fields are optional: `name`, `manufacturer`, `specs`, `description`.
Missing facts are reported; they do not block generation.

The response contains the original `product`, a generated `draft`,
`missing_fields`, and quality `warnings`. Technical characteristics are copied
from the supplied specifications. The model writes the prose and search queries.
Review its factual claims before using a draft.

If the source is too sparse to support the requested description length, the
draft is returned with a warning. The service does not pad it with invented facts.
Missing configuration returns HTTP 503; provider/output failures return HTTP 502.

## Check

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Automated tests use fake clients or mocked HTTP. A real model test requires a
working API key and an actual request through `/products`; a passing unit test
does not verify the quality of generated content.
