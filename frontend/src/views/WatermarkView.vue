<script setup>
import { onMounted, ref } from 'vue';
import FileDropzone from '../components/FileDropzone.vue';
import { embedWatermark, v4Capabilities } from '../api/trace.js';
import { createWatermarkForm, watermarkFormData } from '../forms/watermark.js';
import { showAlert } from '../ui-feedback.js';
import { safeImageUrl } from './result-format.js';

const file = ref(null);
const busy = ref(false);
const result = ref(null);
const capabilities = ref({ dinov2: false, lightglue: false });
const form = createWatermarkForm();

onMounted(async () => {
  try { capabilities.value = await v4Capabilities(); } catch (_) {}
});

async function generate() {
  if (!file.value) { showAlert('请选择图片'); return; }
  busy.value = true;
  try { result.value = await embedWatermark(watermarkFormData(file.value, form)); }
  catch (error) { showAlert(error.message); }
  finally { busy.value = false; }
}
</script>

<template>
  <section class="page-content">
    <div class="page-header">
      <div class="page-title">V4 图片保护</div>
      <div class="page-subtitle">生成带源组认证的 V4 图片记录</div>
      <div class="tag-row">
        <span class="tag tag-blue"><i class="ti ti-fingerprint" /> HMAC64 + RS(16,8)</span>
        <span class="tag tag-teal"><i class="ti ti-photo-scan" /> DINOv2 {{ capabilities.dinov2 ? '可用' : '不可用' }}</span>
        <span class="tag tag-amber"><i class="ti ti-route" /> LightGlue {{ capabilities.lightglue ? '可用' : '不可用' }}</span>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-label">上传原图</div>
        <FileDropzone v-model:file="file" label="将原始图片拖移至此" hint="支持 JPG、PNG、WEBP" />
        <button class="btn-primary embed-button" :disabled="busy" @click="generate">
          <i class="ti" :class="busy ? 'ti-loader' : 'ti-shield-lock'" />
          {{ busy ? '正在生成...' : '生成 V4 记录' }}
        </button>
      </div>
      <div class="card">
        <div class="card-label">生成结果</div>
        <template v-if="result">
          <div class="result-row"><span class="result-key">Trace ID</span><span class="result-val">{{ result.trace_id }}</span></div>
          <div class="result-row"><span class="result-key">状态</span><span class="result-val">{{ result.outcome }}</span></div>
          <div class="result-actions">
            <a v-if="safeImageUrl(result.output_access_url)" class="result-link" :href="safeImageUrl(result.output_access_url)" target="_blank" rel="noopener">打开保护图</a>
          </div>
        </template>
        <div v-else class="result-row"><span class="result-key">状态</span><span class="result-val">-</span></div>
      </div>
    </div>
  </section>
</template>
