# LLM-backed methodology demo agent — a FastAPI service speaking the ATP HTTP
# contract, delegating the task to an OpenAI-compatible model.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

EXPOSE 8000
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
