# VEYRA ATLAS frontend

The Next.js application for uploading SAR scenes, visualizing prediction and flood-analysis layers, and showing live model status.

Next.js dashboard for the SAR Colorization & Disaster Intelligence FastAPI service.

## Run

1. `cd web && npm install`
2. Copy `.env.local.example` to `.env.local` and set `NEXT_PUBLIC_API_BASE_URL` if the FastAPI server is not at `http://127.0.0.1:8000`.
3. Start the FastAPI backend separately (its CORS configuration is already permissive).
4. Run `npm run dev` and visit `http://localhost:3000`.

The UI validates all backend JSON payloads with Zod, uses React Query for API state, and uses MapLibre GL for GeoJSON flood boundaries.
