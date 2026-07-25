import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse, FileResponse, RedirectResponse


app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.get("/new-path")
async def new_path():
    return {"message":"这是新接口"}

@app.get("/old_path")
async def old_path():
    return RedirectResponse(url="/new-path",status_code=307)

@app.get("/download/excel")
async def download_excel():
    excel_path = ""
    return FileResponse(
        path=excel_path,
        filename="",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
@app.get("/api/user")
async def api_user():
    return JSONResponse(
        content={"name":"zhang3","age":20},
        status_code=200, # 可选，默认 200
        headers={"a":'b'} # 可选，自定义响应头
    )

@app.post("/items")
def create_item(item: Item):
    print("create_item后端接口被访问...")
    return {"item":item}


@app.get("/read_root")
def read_root():
    print("read_root后端接口被访问...")
    print("read_root后端接口被访问...")
    print("read_root后端接口被访问...")

    return {"root": "root"}

@app.get("/item/{item_id}")
def read_item(item_id: int,a:str):
    print("read_item后端有参数接口被访问...")
    return {"item_id":item_id,"a":a}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
