FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent.py .

# Make executable
RUN chmod +x agent.py

# ATP Protocol: stdin -> stdout, events -> stderr
ENTRYPOINT ["python", "agent.py"]
