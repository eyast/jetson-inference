from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    with open("position.txt", "r") as f:
        pos = f.read()
        pos = int(pos)
    return pos
    # return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}