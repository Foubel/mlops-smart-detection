FROM python:3.11-slim

WORKDIR /mlflow/

RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql

ENTRYPOINT ["mlflow", "server", "--serve-artifacts","--host","0.0.0.0"]
