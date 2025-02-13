# we use as base image python 3.11-slim
FROM python:3.11-slim

# Set the working directory
WORKDIR /src

RUN pip install fastapi uvicorn pydantic jinja2 praw python-multipart

# copy the rest of the files
COPY ./src .

CMD ["uvicorn", "vkt-app:app", "--host", "0.0.0.0", "--port", "5050"]
