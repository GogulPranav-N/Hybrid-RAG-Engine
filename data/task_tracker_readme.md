# Task Tracker API

A RESTful API built with FastAPI for managing tasks and projects. Containerized with Docker for easy deployment.

## Features

- CRUD operations for tasks (Create, Read, Update, Delete)
- Task categorization with tags and priority levels
- User assignment and due date tracking
- Filter and search tasks by status, priority, or assignee
- SQLite database with SQLAlchemy ORM
- Docker containerized for easy deployment
- Automated testing with pytest

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite + SQLAlchemy
- **Validation**: Pydantic v2
- **Server**: Uvicorn (ASGI)
- **Testing**: pytest + httpx
- **Containerization**: Docker + Docker Compose

## Getting Started

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/task-tracker-api.git
cd task-tracker-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

### Docker

```bash
# Build and run
docker compose up --build

# The API is available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## API Endpoints

A task tracker API typically uses FastAPI with routes for creating tasks (POST /tasks), reading tasks (GET /tasks and GET /tasks/{id}), updating tasks (PUT /tasks/{id}), and deleting tasks (DELETE /tasks/{id}). Each route uses Pydantic models for validation.

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tasks` | List all tasks (with filtering) |
| POST | `/tasks` | Create a new task |
| GET | `/tasks/{id}` | Get task by ID |
| PUT | `/tasks/{id}` | Update a task |
| DELETE | `/tasks/{id}` | Delete a task |
| PATCH | `/tasks/{id}/status` | Update task status |

### Query Parameters for GET /tasks

- `status`: Filter by status (todo, in_progress, done)
- `priority`: Filter by priority (low, medium, high)
- `assignee`: Filter by assigned user
- `tag`: Filter by tag
- `skip`: Pagination offset (default: 0)
- `limit`: Pagination limit (default: 20)

## Data Models

### Task

```python
class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    status: TaskStatus = TaskStatus.TODO
    priority: Priority = Priority.MEDIUM
    assignee: str | None = None
    due_date: date | None = None
    tags: list[str] = []

class TaskResponse(TaskCreate):
    id: int
    created_at: datetime
    updated_at: datetime
```

### Enums

```python
class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

## Project Structure

```
task-tracker-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── models.py         # SQLAlchemy models
│   ├── schemas.py        # Pydantic schemas
│   ├── database.py       # Database connection
│   ├── crud.py           # CRUD operations
│   └── routers/
│       └── tasks.py      # Task routes
├── tests/
│   ├── test_tasks.py
│   └── conftest.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DATABASE_URL | Database connection string | sqlite:///./tasks.db |
| HOST | Server host | 0.0.0.0 |
| PORT | Server port | 8000 |
| LOG_LEVEL | Logging level | info |

## Testing

```bash
# Run tests
pytest -v

# Run with coverage
pytest --cov=app tests/
```

## License

MIT
