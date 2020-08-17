FROM python:3.7

COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

ENV PORT 8080
EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app