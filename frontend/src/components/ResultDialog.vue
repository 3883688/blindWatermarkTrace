<script setup>
import { nextTick, ref, watch } from 'vue';
const props = defineProps({ open: Boolean });
const emit = defineEmits(['close']);
const dialog = ref();
watch(() => props.open, async value => { await nextTick(); if (value) dialog.value?.showModal(); else dialog.value?.close(); });
</script>
<template><dialog ref="dialog" class="result-dialog" @cancel.prevent="emit('close')" @close="emit('close')" @click.self="emit('close')"><div class="dialog-head"><div class="dialog-title"><i class="ti ti-circle-check"></i> 水印处理成功</div><button class="dialog-close" aria-label="关闭" @click="emit('close')"><i class="ti ti-x"></i></button></div><div class="dialog-body">水印已生成，详细数据已展示在上传图片下方。</div></dialog></template>
