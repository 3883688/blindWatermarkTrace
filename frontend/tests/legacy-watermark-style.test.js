import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('watermark view retains shared style selectors with V4-only status markup', async () => {
  const [css, index, tokens, view] = await Promise.all([
    readFile('src/styles/legacy.css', 'utf8'),
    readFile('src/styles/index.css', 'utf8'),
    readFile('src/styles/tokens.css', 'utf8'),
    readFile('src/views/WatermarkView.vue', 'utf8'),
  ]);

  expect(css).toContain('.dialog-title{display:flex;align-items:center;gap:8px;font-size:15px;font-weight:500;color:#5eead4}');
  expect(css).toContain('.embed-result{flex:0 0 100%;margin-top:8px}');
  expect(css).toContain('.login-subtitle{font-size:13px;color:rgba(255,255,255,0.42);margin-bottom:22px}');
  expect(css).toContain('.method-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}');
  expect(css).toContain('.method-name{font-size:13px;color:rgba(255,255,255,0.7)}');
  expect(css).toContain('.preview-size{font-size:11px;color:rgba(255,255,255,0.35)}');
  expect(css).toContain('.progress-wrap{margin-top:8px}');
  expect(css).toContain('.role-save-btn{min-height:34px;white-space:nowrap;flex-shrink:0;align-self:center}');
  expect(css).toContain('.img-name{font-size:13px;font-weight:500;color:rgba(255,255,255,0.85)}');
  expect(view).toContain('HMAC64 + RS(16,8)');
  expect(view).toContain("capabilities.dinov2 ? '可用' : '不可用'");
  expect(view).toContain("capabilities.lightglue ? '可用' : '不可用'");
  expect(view).not.toContain('频域');
  expect(tokens).not.toContain('button, input, select { font: inherit; }');
  expect(tokens).not.toContain('--font-sans');
  expect(index).toContain('.field-range{display:block}');
  expect(index).toContain('.dz-icon{line-height:1}');
});
