export const V4_CODEC = 'hmac64_rs_16_8_split_repeat_sync_v4';

export function createWatermarkForm() { return { codec: V4_CODEC }; }

export function watermarkFormData(file, form) {
  const data = new FormData();
  data.append('file', file);
  data.append('codec', form.codec || V4_CODEC);
  return data;
}
