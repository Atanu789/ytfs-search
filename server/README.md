# My FastAPI Project

A modern FastAPI project with clean architecture and best practices.

## Features

- ✅ FastAPI with async/await support
- ✅ Pydantic v2 for data validation
- ✅ Clean project structure
- ✅ Environment-based configuration
- ✅ CORS middleware
- ✅ API versioning
- ✅ Comprehensive error handling
- ✅ Security utilities
- ✅ Mock CRUD operations

## Installation

1. Create and activate virtual environment:
```bash
python -m venv fastapi-env
source fastapi-env/bin/activate  # On Windows: fastapi-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

### Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/api/v1/openapi.json

## Project Structure

```
my-fastapi-project/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application factory
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/      # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration settings
│   │   └── security.py        # Security utilities
│   ├── models/                # Database models (SQLAlchemy)
│   ├── schemas/               # Pydantic schemas
│   └── services/              # Business logic
├── tests/                     # Test files
├── requirements.txt
├── .env                       # Environment variables
└── README.md
```

## Available Endpoints

### Items
- `GET /api/v1/items/` - List all items
- `GET /api/v1/items/{item_id}` - Get item by ID
- `POST /api/v1/items/` - Create new item
- `PUT /api/v1/items/{item_id}` - Update item
- `DELETE /api/v1/items/{item_id}` - Delete item

### Users
- `GET /api/v1/users/` - List all users
- `GET /api/v1/users/{user_id}` - Get user by ID
- `POST /api/v1/users/` - Create new user
- `PUT /api/v1/users/{user_id}` - Update user

### Health Check
- `GET /health` - Application health status

## Testing

Run tests with pytest:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
