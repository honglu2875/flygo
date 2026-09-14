//! Node-major [N,B] storage; no edge-by-batch tensor and no atomic float sums.
use crate::Graph;
use rayon::prelude::*;

pub struct Executor {
    pub pool: rayon::ThreadPool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_transpose_preserves_canonical_sum_order() {
        let graph = Graph::new(
            &[0, 2, 2, 3, 6],
            &[1, 3, 0, 0, 1, 2],
            &[0, 0, 1, 1],
            &[1.0, -1.0, 1.0, -1.0],
        )
        .unwrap();
        let executor = Executor::new(2).unwrap();
        let weights = [0.25, -0.5, 1.3, -0.6, 0.9, 0.125];
        let input = [
            1.0, -2.0, 0.5, 3.0, 0.25, 0.0, 4.0, -1.0, 2.0, -2.0, 1.0, 5.0,
        ];
        let packed = executor.transpose_weights(&graph, &weights);
        for batch in [1, 3] {
            let mut expected = vec![0.0; graph.neurons() * batch];
            for (edge, &destination) in graph.dst.iter().enumerate() {
                for b in 0..batch {
                    expected[graph.src[edge] * batch + b] +=
                        weights[edge] * input[destination * batch + b];
                }
            }
            let actual =
                executor.transpose_prepared(&graph, &packed, &input[..expected.len()], batch);
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(
                actual,
                executor.transpose(&graph, &weights, &input[..expected.len()], batch)
            );
        }
    }

    #[test]
    fn zero_rows_preserve_forward_transpose_and_accumulated_edge_gradients() {
        let graph = Graph::new(
            &[0, 2, 2, 3, 6],
            &[1, 3, 0, 0, 1, 2],
            &[0, 0, 1, 1],
            &[1.0, -1.0, 1.0, -1.0],
        )
        .unwrap();
        let executor = Executor::new(2).unwrap();
        let weights = [0.25, -0.5, 1.3, -0.6, 0.9, 0.125];
        let input = [
            0.0, -0.0, 0.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0,
        ];
        let cotangent = [1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0];
        let mut forward = [0.0_f32; 12];
        let mut transpose = [0.0_f32; 12];
        let mut expected_edge = [-0.0, 0.25, -0.25, 3.0, -7.0, -0.0];
        let mut edge = expected_edge;
        for (e, &destination) in graph.dst.iter().enumerate() {
            let source = graph.src[e];
            let mut sum = 0.0;
            for b in 0..3 {
                forward[destination * 3 + b] += weights[e] * input[source * 3 + b];
                transpose[source * 3 + b] += weights[e] * cotangent[destination * 3 + b];
                sum += input[source * 3 + b] * cotangent[destination * 3 + b];
            }
            expected_edge[e] += sum;
        }
        executor.edge_vjp(&graph, &input, &cotangent, 3, &mut edge);
        for (actual, expected) in [
            (
                executor.multiply(&graph, &weights, &input, 3),
                forward.to_vec(),
            ),
            (
                executor.transpose(&graph, &weights, &cotangent, 3),
                transpose.to_vec(),
            ),
            (edge.to_vec(), expected_edge.to_vec()),
        ] {
            for (a, b) in actual.iter().zip(expected) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn zero_skipping_does_not_hide_invalid_products() {
        let graph = Graph::new(&[0, 1, 1], &[1], &[0, 0], &[1.0, 1.0]).unwrap();
        let executor = Executor::new(1).unwrap();
        let input = [0.0, 0.0];
        assert!(executor.multiply(&graph, &[f32::INFINITY], &input, 1)[0].is_nan());
        let mut edge = [0.0];
        executor.edge_vjp(&graph, &input, &[f32::INFINITY, 0.0], 1, &mut edge);
        assert!(edge[0].is_nan());
    }
}

impl Executor {
    pub fn new(threads: usize) -> Result<Self, String> {
        if threads == 0 || threads > 240 {
            return Err("Choose 1..240 CPU threads".into());
        }
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| e.to_string())?,
        })
    }
    pub fn multiply(
        &self,
        graph: &Graph,
        weights: &[f32],
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        self.multiply_rows(&graph.indptr, &graph.src, weights, input, batch)
    }
    /// Same row reductions, evaluating only destinations in a validated readout plan.
    pub(crate) fn multiply_required(
        &self, graph: &Graph, weights: &[f32], input: &[f32], batch: usize, required: &[bool],
    ) -> Vec<f32> {
        match self.sparse_rows(input, batch) {
            Some(active) => self.multiply_active::<true, true>(
                &graph.indptr, &graph.src, weights, input, batch, &active, required),
            None => self.multiply_active::<false, true>(
                &graph.indptr, &graph.src, weights, input, batch, &[], required),
        }
    }
    /// Finite model states often have whole rows zeroed by the rectifier.
    /// Keep dense execution when the extra branch would offer little benefit.
    fn sparse_rows(&self, input: &[f32], batch: usize) -> Option<Vec<bool>> {
        let active: Vec<bool> = self.pool.install(|| {
            input
                .par_chunks(batch)
                .map(|row| row.iter().any(|&x| x != 0.0))
                .collect()
        });
        if active.iter().filter(|&&x| x).count() * 5 < active.len() * 4 {
            Some(active)
        } else {
            None
        }
    }
    fn multiply_rows(
        &self,
        indptr: &[usize],
        neighbors: &[usize],
        weights: &[f32],
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        match self.sparse_rows(input, batch) {
            Some(active) => {
                self.multiply_active::<true, false>(indptr, neighbors, weights, input, batch, &active, &[])
            }
            None => self.multiply_active::<false, false>(indptr, neighbors, weights, input, batch, &[], &[]),
        }
    }
    fn multiply_active<const SKIP_ZERO: bool, const SELECT_ROWS: bool>(
        &self,
        indptr: &[usize],
        neighbors: &[usize],
        weights: &[f32],
        input: &[f32],
        batch: usize,
        active: &[bool],
        required: &[bool],
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        self.pool.install(|| {
            output
                .par_chunks_mut(batch)
                .enumerate()
                .for_each(|(row, out)| {
                    if SELECT_ROWS && !required[row] { return; }
                    let start = indptr[row];
                    let end = indptr[row + 1];
                    for (&source, &weight) in neighbors[start..end].iter().zip(&weights[start..end])
                    {
                        if SKIP_ZERO && !active[source] && weight.is_finite() {
                            continue;
                        }
                        let source = source * batch;
                        for (value, &x) in out.iter_mut().zip(&input[source..source + batch]) {
                            *value += weight * x;
                        }
                    }
                })
        });
        output
    }
    pub fn transpose(
        &self,
        graph: &Graph,
        weights: &[f32],
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        let packed = self.transpose_weights(graph, weights);
        self.transpose_prepared(graph, &packed, input, batch)
    }
    /// Pack once per backward call and reuse across recurrent passes.
    pub fn transpose_weights(&self, graph: &Graph, weights: &[f32]) -> Vec<f32> {
        self.pool.install(|| {
            graph
                .transpose_edges
                .par_iter()
                .map(|&edge| weights[edge])
                .collect()
        })
    }
    /// Weights and neighbors share transpose row order; canonical IDs are unchanged.
    pub fn transpose_prepared(
        &self,
        graph: &Graph,
        weights: &[f32],
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        self.multiply_rows(
            &graph.transpose_indptr,
            &graph.transpose_src,
            weights,
            input,
            batch,
        )
    }
    pub fn edge_vjp(
        &self,
        graph: &Graph,
        input: &[f32],
        cotangent: &[f32],
        batch: usize,
        result: &mut [f32],
    ) {
        // Invalid arithmetic must still propagate (0 * infinity is NaN).
        // The ordinary learner supplies finite inputs, but keep this low-level
        // primitive faithful when diagnosing an overflowing backward pass.
        let finite = self.pool.install(|| {
            input
                .par_iter()
                .chain(cotangent.par_iter())
                .all(|x| x.is_finite())
        });
        let input_active = if finite {
            self.sparse_rows(input, batch)
        } else {
            None
        };
        let cotangent_active = if finite {
            self.sparse_rows(cotangent, batch)
        } else {
            None
        };
        self.pool.install(|| {
            result.par_iter_mut().enumerate().for_each(|(edge, value)| {
                let source = graph.src[edge];
                let dest = graph.dst[edge];
                let mut sum = 0.0;
                if !input_active.as_ref().is_some_and(|mask| !mask[source])
                    && !cotangent_active.as_ref().is_some_and(|mask| !mask[dest])
                {
                    let source = source * batch;
                    let dest = dest * batch;
                    for b in 0..batch {
                        sum += input[source + b] * cotangent[dest + b];
                    }
                }
                // Still add positive zero, preserving signed-zero behavior of
                // the original accumulation into an existing gradient buffer.
                *value += sum;
            })
        });
    }
}
