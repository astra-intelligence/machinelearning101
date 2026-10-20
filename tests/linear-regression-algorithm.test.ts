/**
 * Linear Regression Algorithm Tests
 * 
 * Tests for Linear Regression algorithm (gradient descent optimization)
 * Requirements: 2.3, 4.1, 4.2
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LinearRegression } from '../algorithms/regression/LinearRegression';

describe('LinearRegression Algorithm', () => {
  let lr: LinearRegression;

  beforeEach(() => {
    lr = new LinearRegression({
      learningRate: 0.05,
      epochs: 80,
      normalize: false
    });
  });

  describe('Basic Functionality (1D Regression)', () => {
    it('should train and predict on simple linear relationship (y = 2x + 1)', async () => {
      // Deterministic synthetic points along y = 2x + 1
      const X = [[1], [2], [3], [4], [5]];
      const y = [3, 5, 7, 9, 11];

      await lr.fit(X, y);
      const predictions = lr.predict(X);

      expect(predictions).toHaveLength(5);
      for (let i = 0; i < predictions.length; i++) {
        expect(predictions[i]).toBeCloseTo(y[i], 0);
      }
    });

    it('should generalize to unseen test points', async () => {
      const X_train = [[1], [2], [3], [4], [5]];
      const y_train = [3, 5, 7, 9, 11];

      await lr.fit(X_train, y_train);

      // x = 6 should produce ~ 13 (2*6 + 1)
      const unseen = lr.predict([[6], [0]]);
      expect(unseen[0]).toBeCloseTo(13, 0);
      expect(unseen[1]).toBeCloseTo(1, 0);
    });
  });

  describe('Multi-Feature Regression', () => {
    it('should fit multi-variable linear model (y = 2*x1 + 3*x2)', async () => {
      const lrMulti = new LinearRegression({
        learningRate: 0.05,
        epochs: 100,
        normalize: false
      });

      const X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 1]
      ];
      // y = 2*x1 + 3*x2
      const y = [5, 7, 8, 10, 9];

      await lrMulti.fit(X, y);
      const predictions = lrMulti.predict(X);

      expect(predictions).toHaveLength(5);
      for (let i = 0; i < predictions.length; i++) {
        expect(predictions[i]).toBeCloseTo(y[i], 0);
      }
    });
  });

  describe('Feature Normalization', () => {
    it('should support training with feature normalization enabled', async () => {
      const lrNorm = new LinearRegression({
        learningRate: 0.1,
        epochs: 80,
        normalize: true
      });

      // Data on different scales
      const X = [
        [100, 1],
        [200, 2],
        [300, 3],
        [400, 4]
      ];
      const y = [10, 20, 30, 40];

      await lrNorm.fit(X, y);
      const params = lrNorm.getParameters();

      expect(params.normalized).toBe(true);
      expect(params.featureMeans).toHaveLength(2);
      expect(params.featureStds).toHaveLength(2);

      const predictions = lrNorm.predict(X);
      for (let i = 0; i < predictions.length; i++) {
        expect(predictions[i]).toBeCloseTo(y[i], 0);
      }
    });
  });

  describe('Metrics', () => {
    it('should return empty metrics object before fitting', () => {
      const metrics = lr.getMetrics();
      expect(metrics).toEqual({});
    });

    it('should calculate regression metrics (MSE, RMSE, R2, MAE) after fitting', async () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [3, 5, 7, 9, 11];

      await lr.fit(X, y);
      const metrics = lr.getMetrics();

      expect(metrics.mse).toBeDefined();
      expect(metrics.rmse).toBeDefined();
      expect(metrics.r2Score).toBeDefined();
      expect(metrics.mae).toBeDefined();

      expect(metrics.mse!).toBeGreaterThanOrEqual(0);
      expect(metrics.rmse!).toBeGreaterThanOrEqual(0);
      expect(metrics.mae!).toBeGreaterThanOrEqual(0);
      expect(metrics.r2Score!).toBeGreaterThan(0.9);
      expect(metrics.rmse!).toBeCloseTo(Math.sqrt(metrics.mse!), 4);
    });
  });

  describe('Visualization Data', () => {
    it('should return empty array before training', () => {
      expect(lr.getVisualizationData()).toEqual([]);
    });

    it('should produce 1D regression visualization elements (scatter, residuals, loss)', async () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];

      await lr.fit(X, y);
      const viz = lr.getVisualizationData();

      expect(Array.isArray(viz)).toBe(true);
      expect(viz.length).toBe(3);

      const scatterViz = viz.find(v => v.type === 'scatter2d' && v.data?.line);
      const residualsViz = viz.find(v => v.type === 'scatter2d' && v.data?.residuals);
      const lossViz = viz.find(v => v.type === 'line' && v.data?.name === 'Training Loss');

      expect(scatterViz).toBeDefined();
      expect(residualsViz).toBeDefined();
      expect(lossViz).toBeDefined();
      expect(lossViz!.data.x.length).toBe(80);
    });

    it('should produce training loss line for multi-dimensional data', async () => {
      const X = [[1, 2], [2, 3], [3, 4]];
      const y = [5, 8, 11];

      await lr.fit(X, y);
      const viz = lr.getVisualizationData();

      expect(viz.length).toBe(1);
      expect(viz[0].type).toBe('line');
    });
  });

  describe('Error Handling', () => {
    it('should throw error when predicting before training', () => {
      expect(() => lr.predict([[1]])).toThrow('Linear Regression must be trained before making predictions');
    });

    it('should throw error when predicting with mismatched feature dimension', async () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];

      await lr.fit(X, y);
      expect(() => lr.predict([[1, 2]])).toThrow();
    });

    it('should reject training on empty dataset', async () => {
      await expect(lr.fit([], [])).rejects.toThrow();
    });

    it('should reject training when X and y lengths do not match', async () => {
      const X = [[1], [2], [3]];
      const y = [1, 2];

      await expect(lr.fit(X, y)).rejects.toThrow();
    });
  });

  describe('Reset', () => {
    it('should reset model state and require re-training', async () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];

      await lr.fit(X, y);
      expect(() => lr.predict([[1]])).not.toThrow();

      lr.reset();
      expect(() => lr.predict([[1]])).toThrow();
      expect(lr.getMetrics()).toEqual({});
      expect(lr.getVisualizationData()).toEqual([]);
      expect(lr.getParameters().weights).toHaveLength(0);
    });
  });
});
