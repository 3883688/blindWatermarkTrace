import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as resultFormat from '../src/views/result-format.js';

const { createAsyncGuard, safeImageUrl, syncConfidence } = resultFormat;

const testDirectory = dirname(fileURLToPath(import.meta.url));
const watermarkView = readFileSync(
  resolve(testDirectory, '../src/views/WatermarkView.vue'),
  'utf8',
);
const traceView = readFileSync(
  resolve(testDirectory, '../src/views/TraceView.vue'),
  'utf8',
);

describe('watermark and trace result contracts', () => {
  test('does not turn absent V4 synchronization evidence into a zero confidence', () => {
    expect(syncConfidence(undefined)).toBe('-');
    expect(syncConfidence(null)).toBe('-');
    expect(syncConfidence(0.875)).toBe('87.5%');
  });

  test('only accepts existing absolute HTTP or application-relative result image links', () => {
    expect(safeImageUrl('https://example.test/image.png')).toBe('https://example.test/image.png');
    expect(safeImageUrl('/uploads/image.png')).toBe('/uploads/image.png');
    expect(safeImageUrl('//evil.example/image.png')).toBe('');
    expect(safeImageUrl('javascript:alert(1)')).toBe('');
    expect(safeImageUrl('data:image/svg+xml,<svg/>')).toBe('');
  });

  test('stops asynchronous view continuations after unmount', () => {
    const guard = createAsyncGuard();
    expect(guard.isActive()).toBe(true);
    guard.dispose();
    expect(guard.isActive()).toBe(false);
  });

  test('presents V4 as the only watermark version', () => {
    expect(watermarkView).toContain("['水印版本', 'V4']");
    expect(watermarkView).toContain('FFT Pilot 同步');
    expect(watermarkView).toContain('全局/局部 DCT 载荷');
    expect(watermarkView).toContain('RS(8,4) 纠错');
    expect(watermarkView).toContain('32-bit HMAC 认证码唯一绑定记录');
    expect(watermarkView).not.toContain('[1,2,3,4]');
    expect(watermarkView).not.toContain('function isV4');
    expect(traceView).not.toContain('function isV4');
    expect(traceView).not.toContain('pHash 匹配');
  });

  test('describes the production visual authentication pipeline', () => {
    expect(traceView).toContain('优先执行 V4 盲认证');
    expect(traceView).toContain('FFT Pilot 同步与校正');
    expect(traceView).toContain('全局/局部 DCT 瓦片提取 64-bit RS 码字');
    expect(traceView).toContain('RS(8,4) 纠错恢复 32-bit HMAC-SHA256 认证码');
    expect(traceView).toContain('由认证码唯一确定记录');
    expect(traceView).toContain('DINOv2 + pgvector');
    expect(traceView).toContain('ORB/RANSAC');
    expect(traceView).toContain('SuperPoint/LightGlue');
    expect(traceView).toContain('解码组内全部 V4 版本');
    expect(traceView).not.toContain('依次尝试 DCT、空间域、DWT、FFT');
  });

  test('only maps an explicit V4 management record into a trace result', () => {
    expect(typeof resultFormat.traceResultFromRecord).toBe('function');
    expect(traceView).toContain('traceResultFromRecord(record)');
    const { traceResultFromRecord } = resultFormat;
    expect(traceResultFromRecord({ robust_watermark_version: 3, trace_id: 'legacy' })).toBeNull();
    expect(traceResultFromRecord({ robust_watermark_version: '4', trace_id: 'string-v4' })).toBeNull();

    const codeRecovery = { authenticated_tiles: 3 };
    const result = traceResultFromRecord({
      robust_watermark_version: 4,
      trace_id: 'v4-record',
      confidence: 91,
      code_recovery: codeRecovery,
      status: 'V4 认证命中',
    });

    expect(result).toMatchObject({
      robust_watermark_version: 4,
      trace_id: 'v4-record',
      confidence: 91,
      code_recovery: codeRecovery,
      status: 'V4 认证命中',
    });
  });
});
