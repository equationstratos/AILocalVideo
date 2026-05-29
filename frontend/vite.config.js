import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// En dev, on proxifie /api vers le backend FastAPI (port 8000) pour éviter
// les soucis de CORS et garder des URLs relatives côté front.
export default defineConfig({
  plugins: [react()],
  server: {
    // Écoute sur toutes les interfaces pour être accessible depuis l'extérieur
    // du VPS (par défaut Vite ne bind que sur 127.0.0.1).
    host: "0.0.0.0",
    port: 5173,
    // Autorise l'accès via l'IP/domaine du VPS (sinon Vite peut bloquer).
    allowedHosts: true,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
