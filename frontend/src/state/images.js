import { computed, ref, watch } from 'vue';

function matchesFilter(record, filter) {
  return !filter || filter === '全部' || record.status === filter;
}

function confidence(record) {
  return Number(record.confidence || record.conf || 0);
}

function sortRows(rows, sort) {
  return [...rows].sort((left, right) => {
    if (sort === 'created_asc') {
      return String(left.created_at || '').localeCompare(String(right.created_at || ''));
    }
    if (sort === 'confidence_desc') return confidence(right) - confidence(left);
    if (sort === 'confidence_asc') return confidence(left) - confidence(right);
    return String(right.created_at || '').localeCompare(String(left.created_at || ''));
  });
}

export function createImageState(initialRecords = [], pageSize = 10) {
  const records = ref(initialRecords);
  const activeFilter = ref('全部');
  const search = ref('');
  const mode = ref('');
  const date = ref('');
  const sort = ref('created_desc');
  const currentPage = ref(1);
  const selectedIds = ref(new Set());
  const normalizedSearch = computed(() => search.value.trim().toLocaleLowerCase());
  const filteredRows = computed(() => sortRows(records.value.filter(record => {
    const haystack = [
      record.name,
      record.user_id,
      record.trace_id,
      record.mode_label,
      record.mode,
      record.status,
    ].map(value => String(value || '').toLocaleLowerCase()).join(' ');
    if (!matchesFilter(record, activeFilter.value)) return false;
    if (mode.value && record.mode !== mode.value && record.mode_label !== mode.value) return false;
    if (date.value && String(record.created_at || '').slice(0, 10) !== date.value) return false;
    return !normalizedSearch.value || haystack.includes(normalizedSearch.value);
  }), sort.value));
  const totalPages = computed(() => Math.max(1, Math.ceil(filteredRows.value.length / pageSize)));
  const pagedRows = computed(() => {
    const page = Math.min(Math.max(currentPage.value, 1), totalPages.value);
    if (currentPage.value !== page) currentPage.value = page;
    const start = (page - 1) * pageSize;
    return filteredRows.value.slice(start, start + pageSize);
  });

  watch([activeFilter, search, mode, date, sort], () => { currentPage.value = 1; }, { flush: 'sync' });

  return {
    records,
    get activeFilter() { return activeFilter.value; },
    set activeFilter(value) { activeFilter.value = value; },
    get search() { return search.value; },
    set search(value) { search.value = value; },
    get mode() { return mode.value; },
    set mode(value) { mode.value = value; },
    get date() { return date.value; },
    set date(value) { date.value = value; },
    get sort() { return sort.value; },
    set sort(value) { sort.value = value; },
    get currentPage() { return currentPage.value; },
    set currentPage(value) { currentPage.value = value; },
    selectedIds,
    filteredRows,
    pagedRows,
    totalPages,
    setRecords(nextRecords) {
      records.value = nextRecords;
      const recordIds = new Set(nextRecords.map(record => record.id));
      selectedIds.value = new Set([...selectedIds.value].filter(id => recordIds.has(id)));
      currentPage.value = 1;
    },
  };
}
