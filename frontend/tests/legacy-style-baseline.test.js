import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('complete legacy stylesheet retains the required selector baseline and exact key rules', async () => {
  const css = await readFile('src/styles/legacy.css', 'utf8');
  const selectors = [...css.matchAll(/(?:^|\n)([^@\n][^{]+)\{/g)].map(([, selector]) => selector.trim());

  expect(selectors).toEqual(expect.arrayContaining([
    '.advanced-section', '.dropzone', '.embed-button', '.side-panel', '.result-box',
    '.result-row', '.result-link', '.upload-preview', '.confidence-bar', '.icon-btn',
  ]));
  expect(css).toContain('.advanced-section{padding:16px;background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:14px}');
  expect(css).toContain('.dropzone{border:2px dashed rgba(255,255,255,0.12);border-radius:14px;padding:48px;text-align:center;cursor:pointer;transition:all .2s;background:rgba(255,255,255,0.02)}');
  expect(css).toContain('.result-link{display:inline-flex;align-items:center;gap:6px;padding:9px 13px;border-radius:8px;background:rgba(99,102,241,0.28);border:1px solid rgba(129,140,248,0.58);color:#eef2ff;text-decoration:none;font-size:13px;font-weight:600;transition:background .2s,border-color .2s,color .2s}');
});
