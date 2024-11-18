import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

if os.path.isdir("models"):
  MODEL_PATH = "models/linear_model_nb_v1.npy"                
elif os.path.isdir("../models"):
  MODEL_PATH = "../models/linear_model_nb_v1.npy"
else:
  raise Exception("Could not find the models directory")


def log_message(message):
  print("MYLOG:", message, flush=True)

# now we load the linear model
import numpy as np
model_data = np.load(MODEL_PATH, allow_pickle=True).item()
log_message(f"LOADED MODEL DATA: {model_data}")



class AddressInputData(BaseModel):
  nr_cam: int
  mp: int
  parter: int
  et12: int
  et3 : int
  etaj_max : int
  typ_decom : int
  bloc_nou : int
  
app = FastAPI()

def dummy_model(data: AddressInputData):
  inferred_price = (
    data.nr_cam * 10 +
    data.mp * 10 +
    data.parter * 10 +
    data.et12 * 10 +
    data.et3 * 10 +
    data.etaj_max * 10 +
    data.typ_decom * 10 +
    data.bloc_nou * 10
  )
  return inferred_price


def linear_model(data: AddressInputData):
  input_data = []
  for field in model_data["field_mapping"]:
    input_data.append(getattr(data, field))
  np_x = np.array(input_data).reshape(1, -1)
  log_message(f"Input data: {np_x}")
  np_x_mean, np_x_std = model_data["input_transform"]
  np_y_min, np_y_max = model_data["output_transform"]
  np_x_n = (np_x - np_x_mean) / np_x_std
  np_x_nb = np.concatenate([np_x_n, np.ones((np_x_n.shape[0], 1))], axis=1)
  np_w = model_data["weights"]
  log_message(f"Normalized with bias inputs: {np_x_nb}")
  np_yh = np_x_nb.dot(np_w)
  inferred_price = np_yh * (np_y_max - np_y_min) + np_y_min
  log_message(f"Inferred price: {inferred_price}")
  return round(inferred_price[0],2)


# new we prepare the root HTML page
@app.get("/", response_class=HTMLResponse)
async def show_html(request: Request):
  """
  This could have been replaced by:
  
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

  """
  with open("index.html") as f:
    return HTMLResponse(content=f.read(), status_code=200)
    
# the appraise endpoint will receive the data from the form and return the inferred price
@app.post("/appraise")
async def appraise(data:  AddressInputData):
  """
  This endpoint receives the data from the form and returns the inferred price
  """
  inferred_price = linear_model(data)
  return {"inferred_price": inferred_price}