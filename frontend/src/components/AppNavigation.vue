<script setup>
import { ref } from 'vue';

const props = defineProps({
  activePage: { type: String, required: true },
  currentUser: { type: Object, default: null },
  menus: { type: Array, required: true },
  theme: { type: String, required: true },
});

const emit = defineEmits(['select-page', 'update:theme', 'logout']);
const menuOpen = ref(false);
const logoUrl = `${import.meta.env.BASE_URL}site-logo.png`;

const menuItems = [
  { key: 'watermark', label: '生成水印', icon: 'ti-droplet' },
  { key: 'trace', label: '图片溯源', icon: 'ti-route' },
  { key: 'manage', label: '图片管理', icon: 'ti-layout-grid' },
  { key: 'role', label: '角色管理', icon: 'ti-users' },
];

</script>

<template>
  <nav class="navbar">
    <div class="nav-brand">
      <div class="logo-icon"><img :src="logoUrl" alt=""></div>
      <span class="nav-brand-copy" data-brand-title="图片溯源系统（Watermark System）"><span class="nav-brand-cn">图片溯源系统</span><span class="nav-brand-en">（Watermark System）</span></span>
    </div>
    <button
      class="nav-menu-toggle"
      type="button"
      aria-label="打开菜单"
      :aria-expanded="menuOpen"
      title="打开菜单"
      @click="menuOpen = !menuOpen"
    ><i class="ti" :class="menuOpen ? 'ti-x' : 'ti-menu-2'" aria-hidden="true"></i></button>
    <div class="nav-links" :class="{ 'is-open': menuOpen }">
      <button
        v-for="item in menuItems"
        :key="item.key"
        class="nav-link"
        :class="{ active: activePage === item.key }"
        :data-menu="item.key"
        :hidden="!menus.includes(item.key)"
        @click="emit('select-page', item.key); menuOpen = false"
      ><i :class="`ti ${item.icon}`" aria-hidden="true"></i> {{ item.label }}</button>
    </div>
    <div class="nav-right">
      <el-select
        class="app-select theme-select"
        aria-label="主题"
        size="small"
        popper-class="app-select-dropdown"
        :model-value="theme"
        @update:model-value="emit('update:theme', $event)"
      >
        <el-option label="Dark" value="dark" />
        <el-option label="Light" value="light" />
      </el-select>
      <div class="user-pill">
        <div class="user-avatar">BX</div>
        <span class="current-user-name">{{ currentUser?.username || '未登录' }}</span>
      </div>
      <button class="btn-logout" @click="emit('logout')"><i class="ti ti-logout" aria-hidden="true" style="font-size:14px"></i><span class="logout-label">退出登录</span></button>
    </div>
  </nav>
</template>
