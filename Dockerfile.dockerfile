FROM python:3.10

EXPOSE 8050

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ADD requirements.txt
RUN python -m pip install -r requirements.txt