import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  base: '/assets/app/',
  plugins: [vue()],
  build: {
    outDir: '../assets/app',
    emptyOutDir: true,
    assetsDir: '',
    rollupOptions: {
      input: 'src/main.js',
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'app.[ext]',
      },
    },
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/auth': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
  },
});
