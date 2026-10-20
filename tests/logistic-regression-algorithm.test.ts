/**
 * Logistic Regression Algorithm Tests
 * 
 * Tests for binary Logistic Regression algorithm (gradient descent optimization)
 * Requirements: 2.3, 3.1, 4.1
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LogisticRegression } from '../algorithms/classification/LogisticRegression';

describe('LogisticRegression Algorithm', () => {
  let lr: LogisticRegression;

  beforeEach(() => {
    lr = new LogisticRegression({
      learningRate: 0.1,
      epochs: 80,
      normalize: false,
      regularization: 0
    });
  });

  describe('Basic Classification Functionality', () => {
    it('should train and predict on linearly separable 2D data', async () => {
      // Two well-separated clusters
      const X = [
        [0.5, 0.5],
        [0.8, 1.0],
        [1.0, 0.6],
        [5.0, 5.0],
        [5.2, 5.5],
        [5.5, 5.1]
      ];
      const y = [0, 0, 0, 1, 1, 1];

      await lr.fit(X, y);
      const predictions = lr.predict(X);

      expect(predictions).toHaveLength(6);
      expect(predictions).toEqual(y);
    });

    it('should classify unseen test points correctly', async () => {
      const X = [
        [1, 1],
        [1.5, 1.2],
        [6, 6],
        [6.5, 6.2]
      ];
      const y = [0, 0, 1, 1];

      await lr.fit(X, y);

      const testPoints = [[0.8, 0.9], [7.0, 7.0]];
      const predictions = lr.predict(testPoints);

      expect(predictions[0]).toBe(0);
      expect(predictions[1]).toBe(1);
    });

    it('should output calibrated probabilities in range [0, 1]', async () => {
      const X = [
        [1, 1],
        [2, 2],
        [7, 7],
        [8, 8]
      ];
      const y = [0, 0, 1, 1];

      await lr.fit(X, y);
      const probas = lr.predictProba(X);

      expect(probas).toHaveLength(4);
      probas.forEach(p => {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      });

      // Class 0 points should have low p, Class 1 points should have high p
      expect(probas[0]).toBeLessThan(0.5);
      expect(probas[3]).toBeGreaterThan(0.5);
    });
  });

  describe('Regularization and Normalization', () => {
    it('should train successfully with L2 regularization', async () => {
      const lrReg = new LogisticRegression({
        learningRate: 0.1,
        epochs: 60,
        regularization: 0.1,
        normalize: false
      });

      const X = [
        [0, 0],
        [1, 0],
        [4, 4],
        [5, 4]
      ];
      const y = [0, 0, 1, 1];

      await lrReg.fit(X, y);
      const predictions = lrReg.predict(X);
      expect(predictions).toEqual(y);
    });

    it('should train successfully with feature normalization', async () => {
      const lrNorm = new LogisticRegression({
        learningRate: 0.1,
        epochs: 60,
        normalize: true,
        regularization: 0
      });

      // Unnormalized scale differences
      const X = [
        [1000, 0.01],
        [1100, 0.02],
        [5000, 0.8],
        [5200, 0.9]
      ];
      const y = [0, 0, 1, 1];

      await lrNorm.fit(X, y);
      const params = lrNorm.getParameters();
      expect(params.normalized).toBe(true);

      const predictions = lrNorm.predict(X);
      expect(predictions).toEqual(y);
    });
  });

  describe('Metrics', () => {
    it('should return empty metrics before fitting', () => {
      expect(lr.getMetrics()).toEqual({});
    });

    it('should calculate classification metrics (accuracy, precision, recall, f1, confusion matrix)', async () => {
      const X = [
        [1, 1],
        [2, 2],
        [6, 6],
        [7, 7]
      ];
      const y = [0, 0, 1, 1];

      await lr.fit(X, y);
      const metrics = lr.getMetrics();

      expect(metrics.accuracy).toBeDefined();
      expect(metrics.accuracy).toBe(1);
      expect(metrics.precision).toBe(1);
      expect(metrics.recall).toBe(1);
      expect(metrics.f1Score).toBe(1);

      expect(metrics.confusionMatrix).toBeDefined();
      expect(metrics.confusionMatrix).toEqual([
        [2, 0],
        [0, 2]
      ]);
    });
  });

  describe('Visualization Data', () => {
    it('should return empty array before training', () => {
      expect(lr.getVisualizationData()).toEqual([]);
    });

    it('should return scatter plot, heatmap, loss and accuracy curves for 2D classification', async () => {
      const X = [
        [1, 1],
        [2, 2],
        [6, 6],
        [7, 7]
      ];
      const y = [0, 0, 1, 1];

      await lr.fit(X, y);
      const viz = lr.getVisualizationData();

      expect(Array.isArray(viz)).toBe(true);
      expect(viz.length).toBeGreaterThanOrEqual(3);

      const scatter = viz.find(v => v.type === 'scatter2d' && v.data?.class0 && v.data?.class1);
      const heatmap = viz.find(v => v.type === 'heatmap');
      const lossCurve = viz.find(v => v.type === 'line' && v.data?.name === 'Training Loss');
      const accuracyCurve = viz.find(v => v.type === 'line' && v.data?.name === 'Training Accuracy');

      expect(scatter).toBeDefined();
      expect(heatmap).toBeDefined();
      expect(lossCurve).toBeDefined();
      expect(accuracyCurve).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should throw error when predicting before training', () => {
      expect(() => lr.predict([[1, 1]])).toThrow('Logistic Regression must be trained before making predictions');
    });

    it('should throw error when predictProba is called before training', () => {
      expect(() => lr.predictProba([[1, 1]])).toThrow('Logistic Regression must be trained before making predictions');
    });

    it('should enforce exactly 2 classes', async () => {
      const X = [[1], [2], [3]];
      const y = [0, 1, 2];

      await expect(lr.fit(X, y)).rejects.toThrow('Logistic Regression requires exactly 2 classes');
    });

    it('should enforce class labels 0 and 1', async () => {
      const X = [[1], [2]];
      const y = [1, 2];

      await expect(lr.fit(X, y)).rejects.toThrow('Labels must be 0 and 1 for binary classification');
    });

    it('should reject dimension mismatch on predict', async () => {
      const X = [[1, 2], [3, 4]];
      const y = [0, 1];

      await lr.fit(X, y);
      expect(() => lr.predict([[1]])).toThrow();
    });
  });

  describe('Reset', () => {
    it('should reset model state and require re-training', async () => {
      const X = [[1, 1], [4, 4]];
      const y = [0, 1];

      await lr.fit(X, y);
      expect(() => lr.predict([[1, 1]])).not.toThrow();

      lr.reset();
      expect(() => lr.predict([[1, 1]])).toThrow();
      expect(lr.getMetrics()).toEqual({});
      expect(lr.getVisualizationData()).toEqual([]);
      expect(lr.getParameters().weights).toHaveLength(0);
    });
  });
});
