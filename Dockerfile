FROM python:3.7

COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

ENV PORT 8080

CMD ["python","script.py"]