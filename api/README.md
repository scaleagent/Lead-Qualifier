
# API Layer

The API layer organizes all FastAPI endpoints into logical groups, separating web interface concerns from business logic.

## Structure

```
api/
├── __init__.py          # Package initialization  
├── webhooks.py          # SMS/WhatsApp webhook endpoints
├── contractors.py       # Contractor management endpoints
├── health.py           # Health check and status endpoints
└── README.md           # This documentation
```

## Design Principles

### Clear Separation
- **API Layer**: HTTP request/response handling, validation, routing
- **Modules**: Business logic, domain services, data processing
- **Repos**: Data access and persistence

### Logical Grouping
- **webhooks.py**: All Twilio webhook handlers
- **contractors.py**: Contractor CRUD and debug endpoints  
- **health.py**: System status and monitoring endpoints

### Consistent Patterns
- Each router file has its own session dependency
- Consistent logging and error handling
- Clear endpoint documentation with docstrings

## Usage

### Including in FastAPI App
```python
from api.webhooks import router as webhooks_router
from api.contractors import router as contractors_router
from api.health import router as health_router

app.include_router(webhooks_router)
app.include_router(contractors_router)  
app.include_router(health_router)
```

### Adding New Endpoints
1. Create new router file in `api/` directory
2. Follow existing patterns for session dependencies
3. Group related endpoints logically
4. Include comprehensive docstrings
5. Add router to main app

## Benefits

- **Maintainability**: Clear separation of concerns
- **Testability**: Easy to unit test individual endpoint groups
- **Scalability**: Simple to add new endpoint categories
- **Team Development**: Different developers can work on different API areas
- **Documentation**: Self-documenting structure with grouped endpoints
