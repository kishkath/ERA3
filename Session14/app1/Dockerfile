FROM python:3.9-slim

WORKDIR /app

# Copy all necessary files
COPY main.py model.py pl_smoll_checkpoint.ckpt requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
