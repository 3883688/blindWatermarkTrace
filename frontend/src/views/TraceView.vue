<script setup>
import { ref } from 'vue';
import FileDropzone from '../components/FileDropzone.vue';
import { extractUpload, extractUrl } from '../api/trace.js';
import { showAlert } from '../ui-feedback.js';

const file = ref(null);
const url = ref('');
const busy = ref(false);
const result = ref(null);

async function detectUpload() {
  if (!file.value) { showAlert('请选择待检测图片'); return; }
  const data = new FormData(); data.append('file', file.value);
  await run(() => extractUpload(data));
}

async function detectUrl() {
  if (!url.value.trim()) { showAlert('请输入图片链接'); return; }
  const data = new FormData(); data.append('url', url.value.trim());
  await run(() => extractUrl(data));
}

async function run(operation) {
  busy.value = true; result.value = null;
  try { result.value = await operation(); }
  catch (error) { showAlert(error.message); }
  finally { busy.value = false; }
}
</script>

<template>
  <section class="page-content">
    <div class="page-header">
      <div class="page-title">V4 图片溯源</div>
      <div class="page-subtitle">通过视觉召回、几何确认和源组认证定位记录</div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-label">待检测图片</div>
        <div class="field-group"><label>映射图片地址</label><input v-model="url" class="field-input" placeholder="https://example.com/photo.jpg"></div>
        <button class="btn-primary trace-button" :disabled="busy" @click="detectUrl">检测地址</button>
        <div class="or-divider">或上传文件</div>
        <FileDropzone v-model:file="file" label="将待溯源图片拖移至此" hint="支持 JPG、PNG、WEBP" compact />
        <button class="btn-primary trace-button" :disabled="busy" @click="detectUpload">检测文件</button>
      </div>
      <div class="card trace-result-card">
        <div class="card-label">认证结果</div>
        <div class="result-row"><span class="result-key">结果</span><span class="result-val">{{ result?.outcome || (busy ? '处理中' : '-') }}</span></div>
        <div class="result-row"><span class="result-key">Trace ID</span><span class="result-val">{{ result?.record?.trace_id || '-' }}</span></div>
        <div class="result-row"><span class="result-key">源组</span><span class="result-val">{{ result?.record?.source_group_id || '-' }}</span></div>
      </div>
    </div>
  </section>
</template>
