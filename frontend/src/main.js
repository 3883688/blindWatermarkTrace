import { createApp } from 'vue';
import { ElButton, ElDialog } from 'element-plus';
import 'element-plus/theme-chalk/base.css';
import 'element-plus/theme-chalk/el-overlay.css';
import 'element-plus/theme-chalk/el-button.css';
import 'element-plus/theme-chalk/el-dialog.css';
import 'element-plus/theme-chalk/el-message-box.css';
import 'element-plus/theme-chalk/dark/css-vars.css';
import App from './App.vue';
import './styles/index.css';

const app = createApp(App);
app.component('ElButton', ElButton);
app.component('ElDialog', ElDialog);
app.mount('#app');
