import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  testDir: './tests/ui',
  outputDir: 'test-results',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    screenshot: 'only-on-failure',
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE }
      : {},
  },
  webServer: {
    // Vite serves the repository shell and committed compiled assets locally.
    // API requests remain intercepted by app.spec.js.
    command: 'npx vite .. --host 127.0.0.1 --port 4173 --config vite.config.js',
    cwd: fileURLToPath(new URL('.', import.meta.url)),
    port: 4173,
    reuseExistingServer: false,
  },
});
