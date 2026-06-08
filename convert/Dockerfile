FROM python:3.11-slim

RUN pip install uv
WORKDIR /workspace/convert
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml
COPY convert_models.py .

WORKDIR /workspace
CMD ["python", "convert/convert_models.py"]
