FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY prognosis ./prognosis
COPY assets ./assets
COPY models ./models
RUN useradd -m app && mkdir /data && chown app /data
USER app
ENV PROGNOSIS_DB_URL=sqlite:////data/prognosis.db
EXPOSE 8000 8501
