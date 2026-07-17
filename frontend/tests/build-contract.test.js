import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from 'vitest';

test('production bundle has a stable FastAPI asset entry', () => {
  expect(
    existsSync(resolve(process.cwd(), '../assets/app/app.js')),
  ).toBe(true);
});
