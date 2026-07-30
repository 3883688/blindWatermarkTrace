export function createWatermarkForm(userId = '') {
  return {
    userId,
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
  };
}

export function watermarkFormData(file, form) {
  const data = new FormData();
  data.append('file', file);
  data.append('user_id', form.userId);
  data.append('copyright_enabled', String(form.copyrightEnabled));
  data.append('copyright_text', form.copyrightText || '\u00a9 QQ:757675150');
  data.append('copyright_opacity', form.copyrightOpacity || '0.16');
  data.append('copyright_complexity', form.copyrightComplexity || 'medium');
  data.append('copyright_irregular_enabled', String(form.copyrightIrregularEnabled));
  data.append('copyright_prominent_corner_enabled', String(form.copyrightProminentCornerEnabled));
  data.append('fidelity_level', form.fidelityLevel || '0.75');
  data.append('robust_watermark_version', form.robustWatermarkVersion);
  data.append('small_crop_trace_enabled', String(form.smallCropTraceEnabled));
  data.append('small_crop_trace_strength', form.smallCropTraceStrength || '1');
  data.append('small_crop_trace_density', form.smallCropTraceDensity || 'high');
  data.append('dot_matrix_trace_enabled', String(form.dotMatrixTraceEnabled));
  data.append('dot_matrix_trace_strength', form.dotMatrixTraceStrength || '0.85');
  return data;
}
