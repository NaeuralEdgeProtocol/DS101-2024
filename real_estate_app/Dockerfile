# we use as base image python 3.11-slim
FROM python:3.11-slim

# Set the working directory
WORKDIR /real_estate_app


# copy requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# delete requirements.txt
RUN rm requirements.txt

# copy the rest of the files

COPY ./src .


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]

