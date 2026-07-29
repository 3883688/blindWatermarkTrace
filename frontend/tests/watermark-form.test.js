import { describe, expect, test } from 'vitest';
import { createWatermarkForm, V4_CODEC, watermarkFormData } from '../src/forms/watermark.js';

describe('V4-only watermark form', () => {
  test('contains only the pinned codec', () => {
    expect(createWatermarkForm()).toEqual({ codec: V4_CODEC });
  });

  test('submits only the image and pinned codec', () => {
    const file = new File(['image'], 'source.png', { type: 'image/png' });
    const data = watermarkFormData(file, createWatermarkForm());
    expect([...data.entries()].map(([key, value]) => [key, value instanceof File ? value.name : value]))
      .toEqual([['file', 'source.png'], ['codec', V4_CODEC]]);
  });
});
