import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from 'vitest';

test('production bundle has a stable FastAPI asset entry', () => {
  expect(
    existsSync(resolve(process.cwd(), '../assets/app/app.js')),
  ).toBe(true);
});

test('FastAPI entrypoint versions V4 frontend assets', () => {
  const html = readFileSync(resolve(process.cwd(), '../index.html'), 'utf8');
  const cssVersion = html.match(/\/assets\/app\/app\.css\?v=([^"']+)/)?.[1];
  const jsVersion = html.match(/\/assets\/app\/app\.js\?v=([^"']+)/)?.[1];
  expect(cssVersion).toBeTruthy();
  expect(jsVersion).toBe(cssVersion);
});
