import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const navigation = readFileSync(
  resolve('src/components/AppNavigation.vue'),
  'utf8',
);
const styles = readFileSync(resolve('src/styles/index.css'), 'utf8');
const imageTable = readFileSync(
  resolve('src/components/ImageTable.vue'),
  'utf8',
);

describe('mobile application shell', () => {
  it('provides a collapsible navigation menu at the phone breakpoint', () => {
    expect(navigation).toContain('class="nav-menu-toggle"');
    expect(navigation).toContain("'is-open': menuOpen");
    expect(styles).toContain('@media (max-width:560px)');
    expect(styles).toContain('.nav-links.is-open{display:flex}');
    expect(styles).toContain(
      '.nav-menu-toggle{-webkit-appearance:none;appearance:none;width:34px;height:34px',
    );
    expect(styles).toContain('-webkit-tap-highlight-color:transparent');
  });

  it('aligns the unwrapped English brand and centers management totals', () => {
    expect(navigation).toContain('<span class="nav-brand-en">Watermark System</span>');
    expect(navigation).not.toContain('（Watermark System）');
    expect(styles).toContain(
      '.nav-brand-copy{display:flex;flex-direction:column;align-items:flex-start',
    );
    expect(styles).toMatch(/\.stats-card \.s-num\{[^}]*text-align:center/);
  });

  it('uses a labeled mobile image list instead of a wide desktop table', () => {
    expect(imageTable).toContain('class="mobile-table-head"');
    expect(imageTable).toContain('data-label="嵌入时间"');
    expect(imageTable).toContain('data-label="溯源状态"');
    expect(imageTable).toContain('data-label="置信度"');
    expect(styles).toContain('.img-table{display:block;min-width:0}');
    expect(styles).toContain('.img-table td[data-label]{grid-column:1/-1;display:grid;');
    expect(styles).toContain('.img-table .status-cell .status-badge{position:static;display:inline-flex;');
    expect(styles).toContain('.img-table .status-cell .status-badge::before{content:\'\';width:6px;');
    expect(styles).toContain('.card:has(.img-table){overflow:visible}');
  });
});
