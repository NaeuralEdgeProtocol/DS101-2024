from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


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
  inferred_price = dummy_model(data)
  return {"inferred_price": inferred_price}