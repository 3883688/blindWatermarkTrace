import { computed, ref, watch } from 'vue';

function matchesFilter(record, filter) {
  return !filter || filter === '全部' || record.status === filter;
}

export function createImageState(initialRecords = [], pageSize = 10) {
  const records = ref(initialRecords);
  const activeFilter = ref('全部');
  const search = ref('');
  const currentPage = ref(1);
  const selectedIds = ref([]);
  const normalizedSearch = computed(() => search.value.trim().toLocaleLowerCase());
  const filteredRows = computed(() => records.value.filter(record => {
    if (!matchesFilter(record, activeFilter.value)) return false;
    if (!normalizedSearch.value) return true;
    return Object.values(record).some(value => String(value ?? '').toLocaleLowerCase().includes(normalizedSearch.value));
  }));
  const totalPages = computed(() => Math.max(1, Math.ceil(filteredRows.value.length / pageSize)));
  const pagedRows = computed(() => {
    const page = Math.min(currentPage.value, totalPages.value);
    const start = (page - 1) * pageSize;
    return filteredRows.value.slice(start, start + pageSize);
  });

  watch([activeFilter, search], () => { currentPage.value = 1; });

  return {
    records,
    get activeFilter() { return activeFilter.value; },
    set activeFilter(value) { activeFilter.value = value; },
    get search() { return search.value; },
    set search(value) { search.value = value; },
    get currentPage() { return currentPage.value; },
    set currentPage(value) { currentPage.value = value; },
    selectedIds,
    filteredRows,
    pagedRows,
    totalPages,
    setRecords(nextRecords) {
      records.value = nextRecords;
      selectedIds.value = [];
      currentPage.value = 1;
    },
  };
}
