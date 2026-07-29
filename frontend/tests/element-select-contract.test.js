import { readFile } from 'node:fs/promises';
import { expect, test } from 'vitest';

test('all application dropdowns use Element Plus selects', async () => {
  const paths = [
    'src/components/AppNavigation.vue',
    'src/components/ImageFilters.vue',
    'src/views/UserView.vue',
    'src/views/WatermarkView.vue',
  ];
  const sources = await Promise.all(paths.map(path => readFile(path, 'utf8')));
  const main = await readFile('src/main.js', 'utf8');

  expect(sources.join('\n')).not.toMatch(/<select\b/i);
  expect(sources.join('\n').match(/<el-select\b/g)).toHaveLength(5);
  expect(main).toContain('ElOption');
  expect(main).toContain('ElSelect');
  expect(main).toContain("element-plus/theme-chalk/el-select.css");
});
