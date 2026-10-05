FROM python:3.13.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv pip install --system --no-cache .

RUN useradd -m app && chown -R app:app /app
USER app

EXPOSE 8765

CMD ["python", "-m", "src.interface", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
