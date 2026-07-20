import { createApp } from 'vue';
import { ElButton, ElDialog, ElOption, ElSelect } from 'element-plus';
import 'element-plus/theme-chalk/base.css';
import 'element-plus/theme-chalk/el-overlay.css';
import 'element-plus/theme-chalk/el-button.css';
import 'element-plus/theme-chalk/el-dialog.css';
import 'element-plus/theme-chalk/el-message-box.css';
import 'element-plus/theme-chalk/el-option.css';
import 'element-plus/theme-chalk/el-select.css';
import 'element-plus/theme-chalk/dark/css-vars.css';
import App from './App.vue';
import './styles/index.css';

const app = createApp(App);
app.component('ElButton', ElButton);
app.component('ElDialog', ElDialog);
app.component('ElOption', ElOption);
app.component('ElSelect', ElSelect);
app.mount('#app');
