from pathlib import Path

import uvicorn
from fastapi import FastAPI, BackgroundTasks , Request
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import get_recent_messages
from utils.sse_utils import create_sse_queue, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str
    is_stream: bool

# 挂载前端页面
@app.get("/chat.html")
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    return FileResponse(chat_html_path)

# agent发送query请求
@app.post("/query")
async def query(query: QueryRequest,background_tasks: BackgroundTasks):
    user_query = query.query
    session_id = query.session_id
    is_stream = query.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id,TASK_STATUS_PROCESSING, is_stream) # 记录进度

    if is_stream:
        background_tasks.add_task(run_query_graph,session_id,user_query,is_stream) # run_query_graph调用工作流
        return {"message":"任务已开始，请耐心等待","session_id":session_id}
    else:
        run_query_graph(session_id,user_query,is_stream)
        answer = get_task_result(session_id,"session_id","")
        return {
            "message":answer,
            "session_id":session_id,
            "answer":answer
        }

# agent发送session_id和SSE长连接stream请求
@app.get("/stream/{session_id}")
async def stream(session_id:str,request:Request):
    print("生成追踪结果")

    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    records = get_recent_messages(session_id, limit=limit)
    items = []
    for r in records:
        items.append({
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "ts": r.get("ts")
        })
    return {"session_id": session_id, "items": items}

def run_query_graph(session_id:str, user_query:str, is_stream:bool):
    print("调用搜索工作流")

    init_state = {
        "original_query":user_query,
        "session_id":session_id,
        "is_stream":is_stream
    }

    workflow = KBQueryWorkflow()
    for chunk in workflow.run(init_state,stream=is_stream):
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
