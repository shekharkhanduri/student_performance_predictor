# Frontend Setup Guide

## Prerequisites

Ensure you have **Node.js** and **npm** installed on your system.

```bash
# Check Node version (should be 16+)
node --version

# Check npm version (should be 8+)
npm --version
```

If not installed, download from https://nodejs.org/

## Installation

From the project root:

```bash
cd frontend
npm install
cd ..
```

This reads `package.json` and installs all dependencies into `node_modules/`.

## Running the Frontend

### Development Mode (with hot reload)

```bash
python run.py frontend
```

This starts the Vite dev server on http://localhost:5173.

Or manually:

```bash
cd frontend
npm run dev
```

### Production Build

```bash
cd frontend
npm run build
```

Outputs to `frontend/dist/` — ready to deploy anywhere.

## Environment Variables

Create a `.env` file in the `frontend/` directory (copy from `.env.example`):

```bash
cp frontend/.env.example frontend/.env
```

Then edit as needed:

```
VITE_API_BASE_URL=http://localhost:8000
```

## Troubleshooting

**"npm not found"**
- Install Node.js from https://nodejs.org/

**"Module not found" errors**
- Run `npm install` to ensure all dependencies are present

**Port 5173 already in use**
- Change port in `frontend/vite.config.js` or kill the process using the port

**API connection errors**
- Ensure backend is running: `python run.py backend`
- Check `VITE_API_BASE_URL` in `.env`

## Notable Dependencies

- **react**: 18.3.1 — UI library
- **react-dom**: 18.3.1 — React rendering
- **vite**: 5.4.14 — Build tool and dev server
- **@vitejs/plugin-react**: React support in Vite

No heavy plotting libraries or extra bloat — just React and Vite.
