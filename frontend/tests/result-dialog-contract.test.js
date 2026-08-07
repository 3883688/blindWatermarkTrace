import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { expect, test } from 'vitest';

test('all application popups use Element Plus instead of native browser dialogs', async () => {
  const [component, manage, main, packageJson, ...views] = await Promise.all([
    readFile(resolve(process.cwd(), 'src/components/ResultDialog.vue'), 'utf8'),
    readFile(resolve(process.cwd(), 'src/views/ManageView.vue'), 'utf8'),
    readFile(resolve(process.cwd(), 'src/main.js'), 'utf8'),
    readFile(resolve(process.cwd(), 'package.json'), 'utf8'),
    ...['components/LoginOverlay.vue', 'views/WatermarkView.vue', 'views/TraceView.vue', 'views/ManageView.vue', 'views/UserView.vue', 'views/RoleView.vue']
      .map(file => readFile(resolve(process.cwd(), `src/${file}`), 'utf8')),
  ]);

  expect(component).toContain('<el-dialog');
  expect(manage).toContain('<el-dialog');
  const elementPlusImports = main
    .match(/import \{([^}]+)\} from 'element-plus'/)?.[1]
    .split(',')
    .map(name => name.trim()) || [];
  expect(elementPlusImports).toEqual(expect.arrayContaining(['ElButton', 'ElDialog']));
  expect(main).not.toContain("import ElementPlus from 'element-plus'");
  expect(main).toContain("import 'element-plus/theme-chalk/el-message-box.css'");
  expect(JSON.parse(packageJson).dependencies).toHaveProperty('element-plus');
  for (const view of views) expect(view).not.toMatch(/\b(?:alert|confirm)\(/);
});
