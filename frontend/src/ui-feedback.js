import { ElMessageBox } from 'element-plus';

export function showAlert(message, title = '提示') {
  return ElMessageBox.alert(String(message || '操作失败'), title, {
    confirmButtonText: '确定',
    type: 'warning',
    autofocus: false,
    closeOnClickModal: false,
  }).catch(() => undefined);
}

export async function showConfirm(message, title = '确认操作') {
  try {
    await ElMessageBox.confirm(String(message), title, {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
      autofocus: false,
      closeOnClickModal: false,
    });
    return true;
  } catch (_) {
    return false;
  }
}
