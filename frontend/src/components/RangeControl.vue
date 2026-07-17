<script setup>
defineProps({ label: String, value: String, min: { default: 0 }, max: { default: 1 }, step: { default: 0.05 } });
const model = defineModel();
</script>
<template><div class="field-group"><div class="range-head"><label>{{ label }}</label><span class="range-value">{{ value }}</span></div><input v-model="model" class="field-range" type="range" :min="min" :max="max" :step="step"></div></template>
