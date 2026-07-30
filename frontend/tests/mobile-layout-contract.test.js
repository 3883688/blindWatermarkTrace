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
});
