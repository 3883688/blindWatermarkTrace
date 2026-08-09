<script setup>
import { computed } from 'vue';
import { formatDateTime, safeImageUrl } from '../views/result-format.js';
const props = defineProps({ rows: Array, selectedIds: { type: Set, required: true } });
const emit = defineEmits(['toggle', 'toggle-page', 'preview', 'download', 'trace', 'delete']);
const pageIds = computed(() => props.rows.map(row => row.id).filter(Boolean));
const selectedCount = computed(() => pageIds.value.filter(id => props.selectedIds.has(id)).length);
const allSelected = computed(() => pageIds.value.length > 0 && selectedCount.value === pageIds.value.length);
function thumbnail(record) { return safeImageUrl(record.thumbnail_access_url || record.thumbnail_url || record.download_access_url || record.download_url || record.original_access_url || record.original_url); }
function preview(record) { return safeImageUrl(record.download_access_url || record.download_url || record.original_access_url || record.original_url || record.thumbnail_access_url || record.thumbnail_url); }
function download(record) { return safeImageUrl(record.download_access_url || record.download_url); }
function showPreview(record) { emit('preview', { url: preview(record), title: record.name }); }
function confidence(record) { return Number(record.confidence || record.conf || 0); }
function badge(status) { return status === '溯源命中' ? 'badge-blue' : status === '泄露预警' ? 'badge-amber' : 'badge-green'; }
</script>
<template>
  <div class="image-list-shell">
    <div v-if="rows.length" class="mobile-table-head">
      <label>
        <input type="checkbox" :checked="allSelected" :indeterminate.prop="selectedCount > 0 && !allSelected" accent-color="#6366f1" @change="emit('toggle-page', $event.target.checked)">
        选择本页
      </label>
      <span>{{ rows.length }} 条</span>
    </div>
    <table class="img-table">
      <thead><tr><th style="width:44px"><input type="checkbox" :checked="allSelected" :indeterminate.prop="selectedCount > 0 && !allSelected" accent-color="#6366f1" @change="emit('toggle-page', $event.target.checked)"></th><th>图片信息</th><th>水印版本</th><th>嵌入时间</th><th>溯源状态</th><th>置信度</th><th>操作</th></tr></thead>
      <tbody>
        <tr v-for="record in rows" :key="record.id">
          <td class="select-cell"><input type="checkbox" :aria-label="`选择 ${record.name}`" :checked="selectedIds.has(record.id)" accent-color="#6366f1" @change="emit('toggle', record.id, $event.target.checked)"></td>
          <td class="info-cell"><div class="img-info"><button class="thumb-open-btn" title="预览图片" :disabled="!preview(record)" @click="showPreview(record)"><img v-if="thumbnail(record)" class="img-thumb" :src="thumbnail(record)" :alt="record.name"></button><div><div class="img-name"><a v-if="preview(record)" class="img-name-link" :href="preview(record)" target="_blank" rel="noopener noreferrer">{{ record.name }}</a><span v-else>{{ record.name }}</span></div><div class="img-meta">{{ record.size }} · {{ record.user_id || '' }} · {{ record.trace_id || '' }}</div></div></div></td>
          <td class="small-cell" data-label="水印版本">V4</td>
          <td class="date-cell" data-label="嵌入时间">{{ formatDateTime(record.created_at || record.time) }}</td>
          <td class="status-cell" data-label="溯源状态"><span class="status-badge" :class="badge(record.status)">{{ record.status === '保护中' ? '保护中' : record.status }}</span></td>
          <td class="confidence-cell" data-label="置信度"><div class="confidence"><div class="confidence-bar"><div class="confidence-fill" :style="{ width: `${confidence(record)}%` }"></div></div><span>{{ confidence(record) }}%</span></div></td>
          <td class="actions-cell" data-label="操作"><div class="action-btns"><button class="icon-btn" title="溯源" @click="emit('trace', record)"><i class="ti ti-route" aria-hidden="true"></i></button><button class="icon-btn" title="下载" :disabled="!download(record)" @click="emit('download', download(record))"><i class="ti ti-download" aria-hidden="true"></i></button><button class="icon-btn danger" title="删除" @click="emit('delete', record.id)"><i class="ti ti-trash" aria-hidden="true"></i></button></div></td>
        </tr>
        <tr v-if="!rows.length" class="empty-row"><td colspan="7" class="empty-table">暂无图片记录</td></tr>
      </tbody>
    </table>
  </div>
</template>
