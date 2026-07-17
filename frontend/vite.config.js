import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  base: '/assets/app/',
  plugins: [vue()],
  resolve: {
    // Vitest is deliberately rooted at the repository so test paths match CI.
    // Keep Vue resolution anchored to this frontend package.
    alias: {
      vue: fileURLToPath(new URL('./node_modules/vue/dist/vue.runtime.esm-bundler.js', import.meta.url)),
    },
  },
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
