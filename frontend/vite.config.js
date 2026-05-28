import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En dev, on proxifie /api vers le backend FastAPI (port 8000) pour éviter
// les soucis de CORS et garder des URLs relatives côté front.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
