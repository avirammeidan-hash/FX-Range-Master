import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@':                   fileURLToPath(new URL('./src',                         import.meta.url)),
      '@trading/hooks':      fileURLToPath(new URL('../../packages/hooks/src',      import.meta.url)),
      '@trading/api-client': fileURLToPath(new URL('../../packages/api-client/src', import.meta.url)),
      '@trading/ui':         fileURLToPath(new URL('../../packages/ui/src',         import.meta.url)),
    },
  },
  server: {
    port: 3001,
    proxy: {
      '/api': {
        target: 'http://localhost:5001',   // Stocks Flask backend
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react':  ['react', 'react-dom', 'react-router-dom'],
          'vendor-query':  ['@tanstack/react-query'],
          'vendor-charts': ['lightweight-charts', 'recharts'],
          'vendor-lucide': ['lucide-react'],
        },
      },
    },
  },
})
