FROM python:3.7 AS preprocess

COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

ENV PORT 8080
EXPOSE 8080

CMD ["python","preprocess.py"]

FROM python:3.7 AS feature_engineering

COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

ENV PORT 8080
EXPOSE 8080

CMD ["python","feature_engineering.py"]

FROM python:3.7 AS model

COPY . app/
WORKDIR /app

RUN pip install -r requirements.txt

ENV PORT 8080
EXPOSE 8080

CMD ["python","script.py"]

