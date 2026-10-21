/**
 * Feature Scaling Algorithm Tests
 * 
 * Tests for normalization (Min-Max) and standardization (Z-Score) preprocessing algorithms
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { FeatureScaling } from '../algorithms/preprocessing/FeatureScaling';

describe('FeatureScaling Algorithm', () => {
  let scaler: FeatureScaling;

  beforeEach(() => {
    scaler = new FeatureScaling();
  });

  describe('Min-Max Normalization', () => {
    it('should scale features to [0, 1] range', () => {
      const data = [
        [0, 10],
        [5, 20],
        [10, 30]
      ];
      scaler.setData(data);

      const result = scaler.normalize();

      expect(result.method).toBe('normalize');
      expect(result.scaled).toEqual([
        [0, 0],
        [0.5, 0.5],
        [1, 1]
      ]);

      expect(result.statistics.originalMin).toEqual([0, 10]);
      expect(result.statistics.originalMax).toEqual([10, 30]);
      expect(result.statistics.scaledMin).toEqual([0, 0]);
      expect(result.statistics.scaledMax).toEqual([1, 1]);
    });

    it('should handle constant feature columns without division by zero', () => {
      const data = [
        [5, 10],
        [5, 20],
        [5, 30]
      ];
      scaler.setData(data);

      const result = scaler.normalize();

      // Constant column should be scaled to 0
      expect(result.scaled.map(row => row[0])).toEqual([0, 0, 0]);
      expect(result.scaled.map(row => row[1])).toEqual([0, 0.5, 1]);
    });
  });

  describe('Z-Score Standardization', () => {
    it('should transform features to have zero mean and unit variance', () => {
      // Symmetrical data around mean
      const data = [
        [-2, 10],
        [0, 20],
        [2, 30]
      ];
      scaler.setData(data);

      const result = scaler.standardize();

      expect(result.method).toBe('standardize');
      expect(result.scaled).toHaveLength(3);

      // Mean should be 0 for both features
      const mean0 = (result.scaled[0][0] + result.scaled[1][0] + result.scaled[2][0]) / 3;
      const mean1 = (result.scaled[0][1] + result.scaled[1][1] + result.scaled[2][1]) / 3;

      expect(mean0).toBeCloseTo(0, 5);
      expect(mean1).toBeCloseTo(0, 5);

      // Statistics check
      expect(result.statistics.originalMean![0]).toBeCloseTo(0, 5);
      expect(result.statistics.originalMean![1]).toBeCloseTo(20, 5);
      expect(result.statistics.scaledMean![0]).toBeCloseTo(0, 5);
      expect(result.statistics.scaledMean![1]).toBeCloseTo(0, 5);
      expect(result.statistics.scaledStd![0]).toBeCloseTo(1, 5);
      expect(result.statistics.scaledStd![1]).toBeCloseTo(1, 5);
    });

    it('should handle zero-variance constant column', () => {
      const data = [
        [7, 100],
        [7, 200],
        [7, 300]
      ];
      scaler.setData(data);

      const result = scaler.standardize();

      // Constant feature standardizes to 0 (since std is 0)
      expect(result.scaled.map(row => row[0])).toEqual([0, 0, 0]);
      expect(result.statistics.originalStd![0]).toBe(0);
    });
  });

  describe('State Management and Reset', () => {
    it('should initialize with constructor data', () => {
      const initial = [[1, 2], [3, 4]];
      const s = new FeatureScaling(initial);
      const res = s.getResult();

      expect(res.original).toEqual(initial);
      expect(res.scaled).toEqual(initial);
      expect(res.method).toBe('none');
    });

    it('should reset scaled data and statistics back to none', () => {
      const data = [[1, 2], [3, 4]];
      scaler.setData(data);
      scaler.normalize();

      expect(scaler.getResult().method).toBe('normalize');

      scaler.reset();
      const afterReset = scaler.getResult();

      expect(afterReset.method).toBe('none');
      expect(afterReset.scaled).toEqual(data);
      expect(afterReset.statistics).toEqual({});
    });
  });

  describe('Error Handling', () => {
    it('should throw error when normalizing empty data', () => {
      expect(() => scaler.normalize()).toThrow('No data to normalize');
    });

    it('should throw error when standardizing empty data', () => {
      expect(() => scaler.standardize()).toThrow('No data to standardize');
    });
  });
});
