import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

const navigation = readFileSync(
  resolve('src/components/AppNavigation.vue'),
  'utf8',
);
const styles = readFileSync(resolve('src/styles/index.css'), 'utf8');

describe('mobile application shell', () => {
  it('provides a collapsible navigation menu at the phone breakpoint', () => {
    expect(navigation).toContain('class="nav-menu-toggle"');
    expect(navigation).toContain("'is-open': menuOpen");
    expect(styles).toContain('@media (max-width:560px)');
    expect(styles).toContain('.nav-links.is-open{display:flex}');
  });

  it('aligns the unwrapped English brand and centers management totals', () => {
    expect(navigation).toContain('<span class="nav-brand-en">Watermark System</span>');
    expect(navigation).not.toContain('（Watermark System）');
    expect(styles).toContain(
      '.nav-brand-copy{display:flex;flex-direction:column;align-items:flex-start',
    );
    expect(styles).toMatch(/\.stats-card \.s-num\{[^}]*text-align:center/);
  });
});
