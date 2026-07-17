<script setup>
import { computed } from 'vue';
const props = defineProps({ currentPage: Number, totalPages: Number });
const emit = defineEmits(['page']);
const pages = computed(() => Array.from({ length: props.totalPages }, (_, index) => index + 1).filter(page => props.totalPages <= 7 || page === 1 || page === props.totalPages || Math.abs(page - props.currentPage) <= 1));
</script>
<template><div class="page-btns"><button class="page-btn" :disabled="currentPage === 1" @click="emit('page', currentPage - 1)">上一页</button><template v-for="(page, index) in pages" :key="page"><button v-if="index && page - pages[index - 1] > 1" class="page-btn" disabled>...</button><button class="page-btn" :class="{ cur: page === currentPage }" @click="emit('page', page)">{{ page }}</button></template><button class="page-btn" :disabled="currentPage === totalPages" @click="emit('page', currentPage + 1)">下一页</button></div></template>
