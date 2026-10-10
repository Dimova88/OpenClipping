FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENCLIPPING_DATA=/data

# ffmpeg + fonts for caption rendering + opencv runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      fonts-dejavu-core \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY openclipping ./openclipping
COPY assets ./assets

EXPOSE 8000
VOLUME ["/data"]

CMD ["python", "-m", "openclipping.web.app"]
