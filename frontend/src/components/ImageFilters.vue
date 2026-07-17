<script setup>
defineProps({ state: { type: Object, required: true }, advanced: Boolean });
const emit = defineEmits(['update:advanced', 'export']);
const filters = [['全部', '全部'], ['保护中', '已保护'], ['泄露预警', '疑似泄露'], ['溯源命中', '溯源命中']];
</script>
<template>
  <div class="table-toolbar"><div class="search-box"><i class="ti ti-search si" aria-hidden="true"></i><input v-model="state.search" placeholder="搜索文件名、用户、Trace ID…"></div><div class="filter-row"><button v-for="[value, label] in filters" :key="value" class="filter-btn" :class="{ on: state.activeFilter === value }" @click="state.activeFilter = value">{{ label }}</button><button class="btn-outline" style="margin-left:4px" @click="emit('update:advanced', !advanced)"><i class="ti ti-adjustments-horizontal" aria-hidden="true"></i> 筛选</button><button class="btn-outline" @click="emit('export')"><i class="ti ti-download" aria-hidden="true"></i> 导出</button></div></div>
  <div class="advanced-filters" :class="{ show: advanced }"><div class="mini-field"><label>水印模式</label><select v-model="state.mode" class="field-select"><option value="">全部模式</option><option value="hybrid">全部算法</option><option value="dct">DCT + 空间域</option><option value="lsb">仅空间域</option><option value="dwt">DWT + 空间域</option><option value="fft">FFT + 空间域</option></select></div><div class="mini-field"><label>嵌入日期</label><input v-model="state.date" class="field-input" type="date"></div><div class="mini-field"><label>排序</label><select v-model="state.sort" class="field-select"><option value="created_desc">最新在前</option><option value="created_asc">最早在前</option><option value="confidence_desc">置信度高到低</option><option value="confidence_asc">置信度低到高</option></select></div></div>
</template>
