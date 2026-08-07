import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('watermark view retains shared style baselines and exposes only the V4 method marker', async () => {
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
  expect(view).toContain('<span class="method-name">V4 认证水印</span><span class="method-tag badge-blue">唯一版本</span>');
  expect([...view.matchAll(/<span class="method-name">/g)]).toHaveLength(1);
  expect(view).not.toContain('DCT 频域');
  expect(view).not.toContain('FFT 傅里叶');
  expect([...view.matchAll(/class="card" style="margin-bottom:24px"/g)]).toHaveLength(3);
  expect([...view.matchAll(/class="advanced-section-head"/g)]).toHaveLength(1);
  expect(view).toContain('<div class="cb-content"><div class="cb-title">启用小面积截图增强</div><div class="cb-desc">');
  expect(view).toContain('<div class="cb-content"><div class="cb-title">启用点阵追溯水印</div><div class="cb-desc">');
  expect(tokens).not.toContain('button, input, select { font: inherit; }');
  expect(tokens).not.toContain('--font-sans');
  expect(index).toContain('.field-range{display:block}');
  expect(index).toContain('.dz-icon{line-height:1}');
});
