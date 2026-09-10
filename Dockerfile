# ==============================================================================
# Financial GraphRAG: Research Dashboard Application Image
# ==============================================================================
FROM python:3.12-slim

LABEL maintainer="Financial GraphRAG Team"
LABEL description="Financial GraphRAG Streamlit Intelligence Dashboard"
LABEL component="web-application"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
