<script setup>
import { computed, onMounted, ref } from 'vue';
import { listRoles, saveRole } from '../api/trace.js';
const roles = ref({}), menus = ref({}), selected = ref({});
const entries = computed(() => Object.entries(roles.value));
async function load() { try { const config = await listRoles(); roles.value = config.roles || {}; menus.value = config.menus || {}; selected.value = Object.fromEntries(Object.entries(roles.value).map(([key, role]) => [key, [...(role.menus || [])]])); } catch (error) { alert(error.message); } }
async function save(key) { try { const config = await saveRole(key, selected.value[key] || []); roles.value = config.roles || {}; menus.value = config.menus || {}; selected.value[key] = [...(roles.value[key]?.menus || [])]; } catch (error) { alert(error.message); } }
onMounted(load);
</script>
<template><section><div class="card"><div class="section-title">菜单权限</div><div class="role-grid"><div v-for="[key, role] in entries" :key="key" class="role-card"><div class="role-head"><div><div class="role-title">{{ role.label || key }}</div><div class="role-key">{{ key }}</div></div><button class="btn-outline role-save-btn" @click="save(key)"><i class="ti ti-device-floppy"></i> 保存</button></div><div class="role-menu-list"><label v-for="(label, menuKey) in menus" :key="menuKey" class="checkbox-row"><input v-model="selected[key]" type="checkbox" :value="menuKey"><div class="cb-content"><div class="cb-title">{{ label }}</div><div class="cb-desc">{{ menuKey }}</div></div></label></div></div></div></div></section></template>
