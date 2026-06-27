FROM python:3.13-slim

WORKDIR /app

# Install p7zip-full for .rar file extraction (includes 7z)
RUN apt-get update && apt-get install -y p7zip-full && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
