FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch/torchaudio (smaller image, no GPU needed for this demo)
RUN pip install --no-cache-dir torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake in the AASIST repo + pretrained checkpoint at build time (not per-request)
RUN git clone -q https://github.com/clovaai/aasist.git /app/aasist_repo && \
    mkdir -p /app/aasist_repo/models/weights && \
    (test -s /app/aasist_repo/models/weights/AASIST.pth || \
     wget -q https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth \
     -O /app/aasist_repo/models/weights/AASIST.pth)

COPY . .

EXPOSE 7860
ENV PORT=7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
