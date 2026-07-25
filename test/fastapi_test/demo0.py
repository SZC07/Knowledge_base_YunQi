import asyncio

import uvicorn
from fastapi import FastAPI
from starlette.responses import StreamingResponse

app = FastAPI()

# 普通的流式方法
async def generate_stream():
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.3)
        yield word

# 流式输出接口
@app.get("/stream")
async def stream_response():
    print("流式输出接口")
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)