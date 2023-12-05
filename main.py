from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from model import compare

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/ml/model")
def read_root(inp1:str,inp2:str):
    ans=compare(inp1,inp2)[0][0]
    print(ans)
    return {"result":ans*100}

