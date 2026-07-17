<script setup>
import { watch } from 'vue';
import AppNavigation from './components/AppNavigation.vue';
import LoginOverlay from './components/LoginOverlay.vue';
import { createAppState } from './state/app.js';

const state = createAppState();

watch(() => state.theme, theme => {
  document.documentElement.dataset.theme = theme;
}, { immediate: true });
</script>

<template>
  <div class="app">
    <AppNavigation
      :active-page="state.activePage"
      :current-user="state.currentUser"
      :menus="state.visibleMenus"
      :theme="state.theme"
      @select-page="state.selectPage"
      @update:theme="state.setTheme"
      @logout="state.clearUser"
    />
    <main class="page active" :id="`page-${state.activePage}`">
      <h1 class="sr-only">WatermarkSystem</h1>
    </main>
    <LoginOverlay v-if="!state.currentUser" @authenticated="state.setUser" />
  </div>
</template>
