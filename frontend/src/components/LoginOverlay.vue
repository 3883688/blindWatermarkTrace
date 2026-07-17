<script setup>
import { ref } from 'vue';
import { login } from '../api/trace.js';
import { showAlert } from '../ui-feedback.js';

const emit = defineEmits(['authenticated']);
const username = ref('');
const password = ref('');
const submitting = ref(false);

async function submit() {
  submitting.value = true;
  try {
    emit('authenticated', await login(username.value.trim(), password.value));
  } catch (cause) {
    showAlert(cause.message || '登录失败');
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div class="login-overlay show">
    <form class="login-box" @submit.prevent="submit">
      <div class="login-logo"><img :src="'/site-logo.png'" alt="图片溯源系统（Watermark System）"></div>
      <div class="login-title">图片溯源系统（Watermark System）</div>
      <div class="field-group">
        <label for="loginUsername">用户名</label>
        <input id="loginUsername" v-model="username" class="field-input" autocomplete="username">
      </div>
      <div class="field-group">
        <label for="loginPassword">密码</label>
        <input id="loginPassword" v-model="password" class="field-input" type="password" autocomplete="current-password">
      </div>
      <button class="btn-primary" :disabled="submitting"><i class="ti ti-login" aria-hidden="true"></i> {{ submitting ? '登录中...' : '登录' }}</button>
    </form>
  </div>
</template>
