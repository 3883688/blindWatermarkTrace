<script setup>
defineProps({ open: Boolean });
const emit = defineEmits(['close']);
</script>
<template><el-dialog :model-value="open" width="min(520px, calc(100vw - 32px))" align-center append-to-body class="element-result-dialog" @update:model-value="!$event && emit('close')"><template #header><div class="element-dialog-title success"><i class="ti ti-circle-check"></i> 水印处理成功</div></template><div class="element-dialog-copy">水印已生成，详细数据已展示在上传图片下方。</div><template #footer><el-button type="primary" @click="emit('close')">确定</el-button></template></el-dialog></template>
