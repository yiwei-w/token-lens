## Prerequisites

- [uv](https://github.com/astral-sh/uv)
- A dedicated GPU is required to run the backend

## Running the Application

### Start the Backend

1. From the `backend/` directory:
   ```bash
   uv run uvicorn main:app --reload
   ```
   This will start the FastAPI server on port 8000.

### Start the Frontend

1. From the `frontend/` directory:
   ```bash
   npm run dev
   ```
   This will start the Next.js development server on port 3000.

