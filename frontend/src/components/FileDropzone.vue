<script setup>
import { onBeforeUnmount, ref } from 'vue';

const props = defineProps({
  label: { type: String, required: true }, hint: { type: String, required: true },
  icon: { type: String, default: 'ti-cloud-upload' }, compact: Boolean,
});
const emit = defineEmits(['update:file']);
const input = ref();
const file = ref(null);
const objectUrl = ref('');
const progress = ref(0);

function revoke() { if (objectUrl.value) URL.revokeObjectURL(objectUrl.value); objectUrl.value = ''; }
function select(selected) {
  if (!selected) return;
  revoke(); file.value = selected; objectUrl.value = URL.createObjectURL(selected); progress.value = 100;
  emit('update:file', selected);
}
function clear() { revoke(); file.value = null; progress.value = 0; if (input.value) input.value.value = ''; emit('update:file', null); }
function drop(event) { event.preventDefault(); select(event.dataTransfer.files[0]); }
function openPreview() { if (objectUrl.value) window.open(objectUrl.value, '_blank'); }
onBeforeUnmount(revoke);
</script>

<template>
  <div class="dropzone" :class="{ compact }" @click="input.click()" @dragover.prevent @drop="drop">
    <div class="dz-icon"><i :class="`ti ${icon}`" aria-hidden="true"></i></div>
    <div class="dz-text">{{ label }}，或 <span>浏览文件</span></div><div class="dz-hint">{{ hint }}</div>
    <input ref="input" type="file" style="display:none" accept="image/*" @change="select($event.target.files[0])">
  </div>
  <div v-if="file" class="upload-preview">
    <button class="preview-thumb-btn" title="打开原图" @click="openPreview"><img class="preview-thumb" :src="objectUrl" alt="上传图片缩略图"></button>
    <div class="preview-info"><div class="preview-name">{{ file.name }}</div><div class="progress-wrap"><div class="progress-bar"><div class="progress-fill" :style="{ width: `${progress}%` }"></div></div></div></div>
    <button class="preview-remove" @click="clear"><i class="ti ti-x" aria-hidden="true"></i></button>
    <slot name="after-preview" />
  </div>
</template>
