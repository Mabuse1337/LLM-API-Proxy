from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import asyncio

app = FastAPI(title="Llama.cpp Thinking Proxy")

LLAMA_URL = "http://localhost:8080/v1"

async def proxy_request(client: httpx.AsyncClient, path: str, method: str, body: dict = None):
    """Forward request to llama.cpp and yield streamed response chunks."""
    url = f"{LLAMA_URL}/{path}"
    if body:
        async with client.stream(method, url, json=body) as resp:
            async for chunk in resp.aiter_text():
                if chunk.strip():
                    yield chunk
    else:
        async with client.stream(method, url) as resp:
            async for chunk in resp.aiter_text():
                if chunk.strip():
                    yield chunk

@app.get("/v1/models")
async def get_models():
    """Proxy /v1/models to llama.cpp."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{LLAMA_URL}/models")
        return JSONResponse(content=resp.json())

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    async with httpx.AsyncClient() as client:  # For non-streaming only
        stream = body.get("stream", False)
        if not stream:
            # Non-streaming: Forward and wrap reasoning_content if present
            resp = await client.post(f"{LLAMA_URL}/chat/completions", json=body)
            data = resp.json()
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and "reasoning_content" in choice["message"]:
                    reasoning = choice["message"].pop("reasoning_content")
                    choice["message"]["content"] = f"<think>\n{reasoning}\n</think>\n{choice['message'].get('content', '')}"
            return JSONResponse(content=data)
        else:
            # Streaming: Accumulate and yield full SSE events with wrapping
            async def stream_with_thinking():
                async with httpx.AsyncClient() as stream_client:
                    event_buffer = ""
                    in_think = False
                    async for chunk in proxy_request(stream_client, "chat/completions", "POST", body):
                        event_buffer += chunk
                        # Split on full SSE events (\n\n delimiter)
                        events = event_buffer.split("\n\n")
                        event_buffer = events.pop()  # Keep partial event
                        for event in events:
                            if event.strip():
                                # Debug: print(f"Raw event: {event[:100]}...")  # Uncomment for logs
                                lines = event.split("\n")
                                data_line = next((line for line in lines if line.startswith("data: ")), None)
                                if data_line:
                                    try:
                                        data_str = data_line[6:].strip()
                                        data = json.loads(data_str)
                                        if "choices" in data and data["choices"]:
                                            delta = data["choices"][0].get("delta", {})
                                            if "reasoning_content" in delta:
                                                reasoning_delta = delta.pop("reasoning_content")
                                                # Open think if needed
                                                if not in_think:
                                                    open_event = f"data: {json.dumps({'choices': [{'delta': {'content': '<think>\n'}}]})}\n\n"
                                                    yield open_event
                                                    # Debug: print("Opened <think>")
                                                    in_think = True
                                                # Yield reasoning delta
                                                reasoning_event = f"data: {json.dumps({'choices': [{'delta': {'content': reasoning_delta}}]})}\n\n"
                                                yield reasoning_event
                                                # Debug: print(f"Yielded reasoning: {reasoning_delta[:50]}...")
                                            elif in_think:
                                                # Close on first non-reasoning
                                                close_event = f"data: {json.dumps({'choices': [{'delta': {'content': '\n</think>'}}]})}\n\n"
                                                yield close_event
                                                # Debug: print("Closed </think>")
                                                in_think = False
                                                # Yield the original event after close
                                                yield event + "\n\n"
                                            else:
                                                # Pass through normal event
                                                yield event + "\n\n"
                                    except json.JSONDecodeError:
                                        # Pass malformed lines through
                                        yield event + "\n\n"
                    # Flush partial buffer
                    if event_buffer.strip():
                        if in_think:
                            close_event = f"data: {json.dumps({'choices': [{'delta': {'content': '\n</think>'}}]})}\n\n"
                            yield close_event
                            # Debug: print("Flushed close")
                        yield event_buffer + "\n\n"
                        # Debug: print("Flushed buffer")
                    # End stream
                    yield "data: [DONE]\n\n"
            return StreamingResponse(stream_with_thinking(), media_type="text/plain")

@app.post("/v1/completions")
async def completions(request: Request):
    # Similar logic for /v1/completions endpoint if you use it
    body = await request.json()
    async with httpx.AsyncClient() as client:
        stream = body.get("stream", False)
        if not stream:
            resp = await client.post(f"{LLAMA_URL}/completions", json=body)
            data = resp.json()
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "reasoning_content" in choice["text"]:
                    reasoning = choice["text"].pop("reasoning_content", "")  # Assuming it's in text
                    choice["text"] = f"<think>\n{reasoning}\n</think>\n{choice['text']}"
            return JSONResponse(content=data)
        else:
            # Streaming fallback—adapt as needed for completions
            async def stream_completions():
                async with httpx.AsyncClient() as stream_client:
                    async for chunk in proxy_request(stream_client, "completions", "POST", body):
                        yield chunk  # Basic proxy; wrap logic similar to above if deltas have reasoning
            return StreamingResponse(stream_completions(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)