/**
 * K-Means Algorithm Tests
 * 
 * Tests for K-Means Clustering algorithm with k-means++ and random initialization
 * Requirements: 2.4, 4.6
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { KMeans } from '../algorithms/clustering/KMeans';

describe('KMeans Algorithm', () => {
  let kmeans: KMeans;

  beforeEach(() => {
    kmeans = new KMeans({
      k: 2,
      maxIterations: 50,
      tolerance: 1e-4,
      initialization: 'kmeans++',
      randomSeed: 42
    });
  });

  describe('Basic Clustering Functionality', () => {
    it('should separate two well-spaced 2D clusters', async () => {
      // Two distinct clusters around (1, 1) and (10, 10)
      const X = [
        [0.9, 1.1],
        [1.0, 0.9],
        [1.1, 1.0],
        [0.8, 1.2],
        [9.9, 10.1],
        [10.0, 9.9],
        [10.1, 10.0],
        [9.8, 10.2]
      ];

      await kmeans.fit(X);
      const assignments = kmeans.predict(X);

      expect(assignments).toHaveLength(8);

      // First 4 samples should belong to one cluster, next 4 to the other
      const clusterA = assignments[0];
      const clusterB = assignments[4];

      expect(clusterA).not.toBe(clusterB);
      expect(assignments.slice(0, 4).every(c => c === clusterA)).toBe(true);
      expect(assignments.slice(4, 8).every(c => c === clusterB)).toBe(true);
    });

    it('should assign unseen test points to nearest centroid', async () => {
      const X = [
        [1, 1],
        [1.2, 0.8],
        [10, 10],
        [10.2, 9.8]
      ];

      await kmeans.fit(X);

      const testPoints = [
        [1.1, 1.1],   // Close to (1, 1)
        [9.9, 10.1]   // Close to (10, 10)
      ];
      const predictions = kmeans.predict(testPoints);

      const trainPredictions = kmeans.predict([[1, 1], [10, 10]]);
      expect(predictions[0]).toBe(trainPredictions[0]);
      expect(predictions[1]).toBe(trainPredictions[1]);
    });
  });

  describe('Initialization Modes', () => {
    it('should support random initialization', async () => {
      const kmeansRandom = new KMeans({
        k: 2,
        maxIterations: 50,
        tolerance: 1e-4,
        initialization: 'random',
        randomSeed: 123
      });

      const X = [
        [0, 0],
        [0.2, 0.1],
        [5, 5],
        [5.1, 4.9]
      ];

      await kmeansRandom.fit(X);
      const centroids = kmeansRandom.getCentroids();

      expect(centroids).toHaveLength(2);
      expect(centroids[0]).toHaveLength(2);
      expect(centroids[1]).toHaveLength(2);
    });

    it('should record iterative cluster history', async () => {
      const X = [
        [1, 1],
        [2, 2],
        [8, 8],
        [9, 9]
      ];

      await kmeans.fit(X);
      const history = kmeans.getClusterHistory();

      expect(Array.isArray(history)).toBe(true);
      expect(history.length).toBeGreaterThan(0);
      expect(history[0]).toHaveProperty('centroids');
      expect(history[0]).toHaveProperty('assignments');
      expect(history[0]).toHaveProperty('inertia');
    });
  });

  describe('Metrics', () => {
    it('should return empty metrics object before fitting', () => {
      expect(kmeans.getMetrics()).toEqual({});
    });

    it('should calculate clustering metrics (inertia, silhouetteScore, clusterSizes)', async () => {
      const X = [
        [1, 1],
        [1.2, 0.8],
        [8, 8],
        [8.2, 7.8]
      ];

      await kmeans.fit(X);
      const metrics = kmeans.getMetrics();

      expect(metrics.inertia).toBeDefined();
      expect(metrics.inertia!).toBeGreaterThanOrEqual(0);
      expect(metrics.silhouetteScore).toBeDefined();
      expect(metrics.silhouetteScore!).toBeGreaterThan(0);
      expect(metrics.iterations).toBeDefined();
      expect(metrics.clusterSizes).toBeDefined();
      expect(metrics.clusterSizes).toHaveLength(2);
      expect(metrics.clusterSizes![0] + metrics.clusterSizes![1]).toBe(4);
    });
  });

  describe('Visualization Data', () => {
    it('should return empty array before training', () => {
      expect(kmeans.getVisualizationData()).toEqual([]);
    });

    it('should provide 2D scatter and line visualization data for 2D clusters', async () => {
      const X = [
        [1, 1],
        [2, 2],
        [8, 8],
        [9, 9]
      ];

      await kmeans.fit(X);
      const viz = kmeans.getVisualizationData();

      expect(Array.isArray(viz)).toBe(true);
      expect(viz.length).toBe(2);

      const scatter = viz.find(v => v.type === 'scatter2d');
      const line = viz.find(v => v.type === 'line');

      expect(scatter).toBeDefined();
      expect(scatter!.data).toHaveProperty('centroids');
      expect(scatter!.data).toHaveProperty('cluster0');
      expect(scatter!.data).toHaveProperty('cluster1');

      expect(line).toBeDefined();
      expect(line!.data.name).toBe('Inertia');
    });
  });

  describe('Error Handling', () => {
    it('should throw error when predicting before training', () => {
      expect(() => kmeans.predict([[1, 1]])).toThrow('K-Means Clustering must be trained before making predictions');
    });

    it('should throw error when k is greater than number of samples', async () => {
      const kLarge = new KMeans({ k: 5, maxIterations: 10, tolerance: 1e-4, initialization: 'kmeans++' });
      await expect(kLarge.fit([[1, 1], [2, 2]])).rejects.toThrow('cannot exceed number of samples');
    });

    it('should throw error when k is less than 1', async () => {
      const kZero = new KMeans({ k: 0, maxIterations: 10, tolerance: 1e-4, initialization: 'kmeans++' });
      await expect(kZero.fit([[1, 1]])).rejects.toThrow('Number of clusters must be at least 1');
    });

    it('should throw error on predict with mismatched feature dimensions', async () => {
      const X = [[1, 2], [3, 4], [5, 6]];
      await kmeans.fit(X);
      expect(() => kmeans.predict([[1, 2, 3]])).toThrow();
    });
  });

  describe('Reset', () => {
    it('should reset model state and centroids', async () => {
      const X = [[1, 1], [2, 2], [5, 5], [6, 6]];
      await kmeans.fit(X);
      expect(() => kmeans.predict([[1, 1]])).not.toThrow();

      kmeans.reset();
      expect(() => kmeans.predict([[1, 1]])).toThrow();
      expect(kmeans.getCentroids()).toHaveLength(0);
      expect(kmeans.getAssignments()).toHaveLength(0);
      expect(kmeans.getMetrics()).toEqual({});
      expect(kmeans.getVisualizationData()).toEqual([]);
    });
  });
});
