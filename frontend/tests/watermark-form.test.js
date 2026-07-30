import { describe, expect, test } from 'vitest';
import { createWatermarkForm, watermarkFormData } from '../src/forms/watermark.js';

describe('watermark form parity', () => {
  test('uses only V4 request defaults', () => {
    const form = createWatermarkForm('BX100');

    expect(form).toMatchObject({
      userId: 'BX100',
      robustWatermarkVersion: '4',
      copyrightEnabled: false,
      copyrightText: '\u00a9 QQ:757675150',
      copyrightIrregularEnabled: true,
      copyrightProminentCornerEnabled: false,
      copyrightOpacity: '0.16',
      copyrightComplexity: 'medium',
      fidelityLevel: '0.75',
      smallCropTraceEnabled: true,
      smallCropTraceStrength: '1',
      smallCropTraceDensity: 'high',
      dotMatrixTraceEnabled: false,
      dotMatrixTraceStrength: '0.85',
    });
  });

  test('creates V4 embed FormData', () => {
    const file = new File(['image'], 'source.png', { type: 'image/png' });
    const data = watermarkFormData(file, createWatermarkForm('BX100'));

    expect([...data.entries()].map(([key, value]) => [key, value instanceof File ? value.name : value])).toEqual([
      ['file', 'source.png'], ['user_id', 'BX100'],
      ['copyright_enabled', 'false'], ['copyright_text', '\u00a9 QQ:757675150'],
      ['copyright_opacity', '0.16'], ['copyright_complexity', 'medium'],
      ['copyright_irregular_enabled', 'true'], ['copyright_prominent_corner_enabled', 'false'],
      ['fidelity_level', '0.75'], ['robust_watermark_version', '4'],
      ['small_crop_trace_enabled', 'true'], ['small_crop_trace_strength', '1'],
      ['small_crop_trace_density', 'high'], ['dot_matrix_trace_enabled', 'false'],
      ['dot_matrix_trace_strength', '0.85'],
    ]);
  });
});
