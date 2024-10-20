
FROM python:3.9-slim

WORKDIR /app

COPY . /app
RUN apt-get update && \
    apt-get install -y python3 python3-pip sqlite3 libsqlite3-dev
RUN pip install -r setup.txt
RUN chmod 666 /app/users.db

EXPOSE 5000

ENV FLASK_APP=main.py

CMD ["flask", "run", "--host", "0.0.0.0"]