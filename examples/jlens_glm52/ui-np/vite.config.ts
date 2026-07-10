import react from '@vitejs/plugin-react';
import path from 'node:path';
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
    },
  },
  server: {
    // Dev-only convenience: proxy the lens API to a local sidecar so `npm run
    // dev` works against a running backend. Production serves dist/ from the
    // same origin as the API, so no proxy is involved there.
    proxy: {
      '/api': {
        target: process.env.JLENS_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
