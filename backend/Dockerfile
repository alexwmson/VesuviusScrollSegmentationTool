FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api/endpoint.py
ENV FLASK_ENV=production
ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN apt-get update && apt-get install -y \
    libtiff6 libjpeg62-turbo libpng-dev gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api_requirements.txt /app/
RUN pip install --no-cache-dir -r api_requirements.txt

COPY . .

EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
