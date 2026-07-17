<script setup>
import { computed, onMounted, ref } from 'vue';
import { deleteImage, listImages } from '../api/trace.js';
import ImageFilters from '../components/ImageFilters.vue';
import ImageTable from '../components/ImageTable.vue';
import Pager from '../components/Pager.vue';
import { createImageState } from '../state/images.js';
import { downloadCsv, setPageSelection } from '../state/management.js';
import { showAlert, showConfirm } from '../ui-feedback.js';
import { safeImageUrl } from './result-format.js';

const emit = defineEmits(['trace']);
const state = createImageState();
const advanced = ref(false), preview = ref(null), stats = ref({});
const selectedIds = state.selectedIds;
const selectedRows = computed(() => state.records.value.filter(row => selectedIds.value.has(row.id)));
const pageIds = computed(() => state.pagedRows.value.map(row => row.id).filter(Boolean));
const count = computed(() => state.filteredRows.value.length);
const start = computed(() => count.value ? (state.currentPage - 1) * 10 + 1 : 0);
const end = computed(() => Math.min(state.currentPage * 10, count.value));
async function load() { try { const response = await listImages(); state.setRecords(response.items || []); stats.value = response.stats || {}; } catch (error) { showAlert(error.message); } }
function toggle(id, checked) { selectedIds.value = setPageSelection(selectedIds.value, [id], checked); }
function togglePage(checked) { selectedIds.value = setPageSelection(selectedIds.value, pageIds.value, checked); }
function exportRecords(selectedOnly = false) { const rows = selectedOnly && selectedRows.value.length ? selectedRows.value : state.filteredRows.value; if (!rows.length) return showAlert('没有可导出的图片记录'); downloadCsv(rows); }
async function remove(id) { if (!(await showConfirm('确认删除这条图片记录？'))) return; try { await deleteImage(id); await load(); } catch (error) { showAlert(error.message); } }
function download(url) { const safeUrl = safeImageUrl(url); if (safeUrl) window.open(safeUrl, '_blank', 'noopener,noreferrer'); }
onMounted(load);
</script>
<template><section class="page-content"><div class="page-header"><div class="page-title">图片管理</div><div class="page-subtitle">管理所有已打水印的图片，查看溯源状态与传播记录</div></div><div class="stats-bar"><div class="stats-card"><div class="s-icon indigo"><i class="ti ti-photo"></i></div><div class="s-num">{{ stats.total ?? 0 }}</div><div class="s-lbl">总图片数</div></div><div class="stats-card"><div class="s-icon teal"><i class="ti ti-shield-check"></i></div><div class="s-num">{{ stats.protected ?? 0 }}</div><div class="s-lbl">已保护</div></div><div class="stats-card"><div class="s-icon amber"><i class="ti ti-alert-circle"></i></div><div class="s-num">{{ stats.leaks ?? 0 }}</div><div class="s-lbl">疑似泄露</div></div><div class="stats-card"><div class="s-icon red"><i class="ti ti-eye"></i></div><div class="s-num">{{ stats.hits ?? 0 }}</div><div class="s-lbl">溯源命中</div></div></div><div class="card"><ImageFilters :state="state" :advanced="advanced" @update:advanced="advanced = $event" @export="exportRecords()"/><div class="selection-bar" :class="{ show: selectedIds.size > 0 }"><span>已选择 {{ selectedIds.size }} 张图片</span><div class="selection-actions"><button class="btn-outline" @click="exportRecords(true)"><i class="ti ti-download"></i> 导出所选</button><button class="btn-outline" @click="selectedIds = new Set()"><i class="ti ti-x"></i> 清空选择</button></div></div><ImageTable :rows="state.pagedRows.value" :selected-ids="selectedIds" @toggle="toggle" @toggle-page="togglePage" @preview="preview = $event" @download="download" @trace="emit('trace', $event)" @delete="remove"/><div class="pagination"><div class="page-info">共 <strong>{{ count }}</strong> 条记录，显示 {{ start }}-{{ end }}，每页 10 条</div><Pager :current-page="state.currentPage" :total-pages="state.totalPages.value" @page="state.currentPage = $event"/></div></div><el-dialog :model-value="Boolean(preview)" width="min(920px, calc(100vw - 32px))" align-center append-to-body class="element-image-dialog" @update:model-value="!$event && (preview = null)"><template #header><div class="element-dialog-title"><i class="ti ti-photo"></i> {{ preview?.title || '图片预览' }}</div></template><div class="element-image-preview"><img v-if="preview" :src="preview.url" :alt="preview.title || '图片预览'"></div></el-dialog></section></template>
