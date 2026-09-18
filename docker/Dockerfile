FROM python:3.9

WORKDIR /app

COPY docker/requirements.txt /app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt && rm requirements.txt

EXPOSE 8888
