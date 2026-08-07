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
  const [navigation, filters, users, watermark] = sources;
  const main = await readFile('src/main.js', 'utf8');
  const imageTable = await readFile('src/components/ImageTable.vue', 'utf8');

  expect(sources.join('\n')).not.toMatch(/<select\b/i);
  // The V4-only UI has five declared dropdowns: theme, image sorting,
  // two user-role controls, and small-crop density.
  expect(navigation.match(/<el-select\b/g)).toHaveLength(1);
  expect(navigation).toContain('aria-label="主题"');
  expect(filters).not.toContain('state.mode');
  expect(filters).not.toMatch(/DCT|DWT|FFT|空间域|全部算法/);
  expect(filters).toContain('<el-select v-model="state.sort"');
  expect(users).toContain('<el-select v-model="createRole"');
  expect(users).toContain('<el-select v-model="choices[name]"');
  expect(watermark).toContain('<el-select v-model="form.smallCropTraceDensity"');
  expect(sources.join('\n').match(/<el-select\b/g)).toHaveLength(6);

  expect(watermark).toContain('水印版本');
  expect(watermark).toContain('value="V4" disabled');
  expect(imageTable).toContain('<th>水印版本</th>');
  expect(imageTable).toContain('<td class="small-cell">V4</td>');
  expect(watermark).not.toContain('v-model="form.robustWatermarkVersion"');
  expect(main).toContain('ElOption');
  expect(main).toContain('ElSelect');
  expect(main).toContain("element-plus/theme-chalk/el-select.css");
});
