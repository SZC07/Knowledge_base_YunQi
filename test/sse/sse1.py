import asyncio

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有的源
    allow_credentials=True, # 允许客户端传递cookie
    allow_methods=["*"], # get和post都行
    allow_headers=["*"], # 请求头所有信息都行
)

async def event_generator():
    for i in range(10):
        yield f"data: 这是第{i}条消息\n\n"
        await asyncio.sleep(1)

@app.get("/simple_stream")
async def simple_stream():
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)
