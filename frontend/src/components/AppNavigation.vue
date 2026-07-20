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

</script>

<template>
  <nav class="navbar">
    <div class="nav-brand">
      <div class="logo-icon"><img :src="'/site-logo.png'" alt=""></div>
      图片溯源系统（Watermark System）
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
      <button class="btn-logout" @click="emit('logout')"><i class="ti ti-logout" aria-hidden="true" style="font-size:14px"></i> 退出登录</button>
    </div>
  </nav>
</template>
