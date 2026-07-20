<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { AUTH_INVALID_EVENT } from './auth-session.js';
import AppNavigation from './components/AppNavigation.vue';
import LoginOverlay from './components/LoginOverlay.vue';
import { createAppState } from './state/app.js';
import WatermarkView from './views/WatermarkView.vue';
import TraceView from './views/TraceView.vue';
import ManageView from './views/ManageView.vue';
import RoleView from './views/RoleView.vue';
import UserView from './views/UserView.vue';

const state = createAppState();
const traceRecord = ref(null);

function clearSession() {
  traceRecord.value = null;
  state.clearUser();
}

onMounted(() => window.addEventListener(AUTH_INVALID_EVENT, clearSession));
onBeforeUnmount(() => window.removeEventListener(AUTH_INVALID_EVENT, clearSession));

function showTraceRecord(record) {
  traceRecord.value = record;
  state.selectPage('trace');
}

watch(() => state.theme, theme => {
  document.documentElement.dataset.theme = theme;
  document.documentElement.classList.toggle('dark', theme === 'dark');
}, { immediate: true });
</script>

<template>
  <div class="app">
    <AppNavigation
      :active-page="state.activePage"
      :current-user="state.currentUser"
      :menus="state.visibleMenus"
      :theme="state.theme"
      @select-page="state.selectPage($event)"
      @update:theme="state.setTheme($event)"
      @logout="clearSession"
    />
    <main v-if="state.currentUser" class="page active" :id="`page-${state.activePage}`">
      <WatermarkView v-if="state.activePage === 'watermark'" :current-user="state.currentUser" />
      <TraceView v-else-if="state.activePage === 'trace'" :record="traceRecord" />
      <ManageView v-else-if="state.activePage === 'manage'" @trace="showTraceRecord" />
      <template v-else-if="state.activePage === 'role'">
        <div class="page-content">
          <div class="page-header">
            <div class="page-title">角色管理</div>
            <div class="page-subtitle">为不同角色分配可访问菜单，仅管理员可见</div>
          </div>
          <UserView />
          <RoleView />
        </div>
      </template>
      <h1 v-else class="sr-only">WatermarkSystem</h1>
    </main>
    <LoginOverlay v-if="!state.currentUser" @authenticated="state.setUser($event)" />
  </div>
</template>
