FROM python:3.12.0

WORKDIR .

COPY requirements.txt .
RUN pip install -r requirements.txt

# delete requirements.txt
RUN rm requirements.txt

# copy the rest of the files

COPY ./backend ./backend
COPY ./model/matrix_file ./model/matrix_file
COPY ./model/model_file ./model/model_file
COPY ./model/sanitized_ingredients.txt ./model/sanitized_ingredients.txt
COPY ./model/sanitized_recipes.json ./model/sanitized_recipes.json

WORKDIR ./backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]
