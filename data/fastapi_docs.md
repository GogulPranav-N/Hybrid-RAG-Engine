# FastAPI Documentation — Quick Reference

## Introduction

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python based on standard Python type hints. It's built on top of Starlette for the web parts and Pydantic for the data parts.

Key features:
- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Type hints enable great editor support and auto-completion
- **Automatic docs**: Interactive API documentation via Swagger UI and ReDoc
- **Standards-based**: Based on OpenAPI and JSON Schema

## Installation

```bash
pip install "fastapi[standard]"
```

This installs FastAPI along with uvicorn as the ASGI server.

## Creating Your First API

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run it with: `uvicorn main:app --reload`

## Path Parameters

You define path parameters by declaring them in the route decorator path string using curly braces, like `@app.get('/items/{item_id}')`, and then including the same name as a function parameter with a type annotation.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

FastAPI automatically validates that `item_id` is an integer and returns a 422 error if it's not.

## Query Parameters

Parameters that are not part of the path are automatically interpreted as query parameters:

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

## Request Body with Pydantic

Pydantic models in FastAPI are used for request body validation, serialization, and documentation. You define a class inheriting from BaseModel with typed fields, and FastAPI automatically validates incoming JSON against it, generates OpenAPI docs, and provides editor support.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Response Models

Use the `response_model` parameter to define the shape of the response:

```python
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    return items[item_id]
```

## Dependency Injection

FastAPI uses the Depends() function for dependency injection. You create a function (or class) that returns a value, then declare it as a parameter with Depends(). FastAPI resolves the dependency tree, supports async dependencies, and handles caching within a request.

```python
from fastapi import Depends

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def read_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

## Middleware

Middleware runs before every request and after every response:

```python
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## Authentication

FastAPI supports authentication through security utilities like OAuth2PasswordBearer, HTTPBasic, and API key headers. You can create middleware or dependency functions that validate tokens, and use the Security() or Depends() mechanism to protect routes.

```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user
```

## Background Tasks

FastAPI BackgroundTasks allow you to run functions after returning a response. You add tasks via background_tasks.add_task(). Use them for non-critical operations like sending emails, logging, or cleanup that shouldn't delay the response to the client.

```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", mode="a") as f:
        f.write(message)

@app.post("/send-notification/")
async def send_notification(
    email: str, background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    return {"message": "Notification sent"}
```

## Streaming Responses

FastAPI supports streaming responses using StreamingResponse. You pass an async generator or iterator that yields chunks of data. For Server-Sent Events, use the sse-starlette library's EventSourceResponse with an async generator.

```python
from fastapi.responses import StreamingResponse

async def generate_large_file():
    for i in range(1000):
        yield f"Line {i}\n"

@app.get("/download")
async def download():
    return StreamingResponse(generate_large_file(), media_type="text/plain")
```

For SSE streaming:

```python
from sse_starlette import EventSourceResponse

async def event_stream(request: Request):
    for i in range(100):
        if await request.is_disconnected():
            break
        yield {"data": f"Event {i}"}
        await asyncio.sleep(0.1)

@app.get("/events")
async def sse(request: Request):
    return EventSourceResponse(event_stream(request))
```

## Environment Variables with pydantic-settings

In FastAPI, you typically handle environment variables using pydantic-settings. You create a Settings class that inherits from BaseSettings, define fields with type annotations and defaults, and it automatically reads from environment variables or a .env file.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My App"
    admin_email: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

## CORS (Cross-Origin Resource Sharing)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing

Use `TestClient` from Starlette for testing:

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```
