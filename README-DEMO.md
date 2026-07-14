# CausalStress Public Demo

Use these steps when you need to expose the local full-stack app over public ngrok URLs for a live demo.

## 1. Start the backend

From the repo root:

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Leave this terminal running.

## 2. Start the frontend

Open a second terminal from the repo root:

```powershell
cd frontend
npm run dev -- --port 8080
```

Leave this terminal running.

## 3. Start ngrok for both apps

Open a third terminal from the repo root:

```powershell
ngrok start --all --config ngrok.yml
```

Ngrok will print two public URLs: one for `frontend` and one for `backend`.

## 4. Point the frontend at the public backend

Copy the `backend` ngrok URL into `frontend/.env.local`:

```ini
VITE_API_URL=https://REPLACE_WITH_YOUR_BACKEND_NGROK_URL
```

Then stop and restart the frontend dev server so Vite reloads the env var:

```powershell
cd frontend
npm run dev -- --port 8080
```

## 5. Send the right URL

Send your manager the `frontend` ngrok URL. The `backend` ngrok URL is only used in `frontend/.env.local` as `VITE_API_URL`.

## CORS note

The backend keeps `http://localhost:8080` for local dev and also accepts ngrok demo origins through a restricted ngrok URL regex. You can set `ALLOWED_ORIGIN` to a single exact frontend ngrok URL before starting the backend if you want a stricter demo origin.
