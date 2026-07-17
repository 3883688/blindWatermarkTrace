<script setup>
const props = defineProps({
  activePage: { type: String, required: true },
  currentUser: { type: Object, default: null },
  menus: { type: Array, required: true },
  theme: { type: String, required: true },
});

const emit = defineEmits(['select-page', 'update:theme', 'logout']);

const menuItems = [
  { key: 'watermark', label: '生成水印', icon: 'ti-droplet' },
  { key: 'trace', label: '图片溯源', icon: 'ti-route' },
  { key: 'manage', label: '图片管理', icon: 'ti-layout-grid' },
  { key: 'role', label: '角色管理', icon: 'ti-users' },
];

const initial = name => name?.slice(0, 2).toUpperCase() || 'BX';
</script>

<template>
  <nav class="navbar">
    <div class="nav-brand">
      <div class="logo-icon"><img :src="'/site-logo.png'" alt=""></div>
      图片溯源系统
    </div>
    <div class="nav-links">
      <button
        v-for="item in menuItems"
        :key="item.key"
        class="nav-link"
        :class="{ active: activePage === item.key }"
        :data-menu="item.key"
        :hidden="!menus.includes(item.key)"
        @click="emit('select-page', item.key)"
      ><i :class="`ti ${item.icon}`" aria-hidden="true"></i> {{ item.label }}</button>
    </div>
    <div class="nav-right">
      <select class="field-select theme-select" aria-label="主题" :value="theme" @change="emit('update:theme', $event.target.value)">
        <option value="dark">Dark</option>
        <option value="light">Light</option>
      </select>
      <div class="user-pill">
        <div class="user-avatar">{{ initial(currentUser?.username) }}</div>
        <span class="current-user-name">{{ currentUser?.username || '未登录' }}</span>
      </div>
      <button class="btn-logout" @click="emit('logout')"><i class="ti ti-logout" aria-hidden="true" style="font-size:14px"></i> 退出登录</button>
    </div>
  </nav>
</template>
