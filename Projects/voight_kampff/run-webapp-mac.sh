cd src
PYTHONPATH=${pwd}/src uvicorn vkt-app:app --reload --port 5050
cd ..