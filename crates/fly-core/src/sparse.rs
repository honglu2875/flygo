//! Node-major [N,B] storage; no edge-by-batch tensor and no atomic float sums.
use crate::Graph;
use rayon::prelude::*;

pub struct Executor {
    pub pool: rayon::ThreadPool,
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
        let mut output = vec![0.0; input.len()];
        self.pool.install(|| {
            output
                .par_chunks_mut(batch)
                .enumerate()
                .for_each(|(row, out)| {
                    for (offset, &w) in weights[graph.indptr[row]..graph.indptr[row + 1]]
                        .iter()
                        .enumerate()
                    {
                        let source = graph.src[graph.indptr[row] + offset] * batch;
                        for (value, &x) in out.iter_mut().zip(&input[source..source + batch]) {
                            *value += w * x;
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
        let mut output = vec![0.0; input.len()];
        self.pool.install(|| {
            output
                .par_chunks_mut(batch)
                .enumerate()
                .for_each(|(row, out)| {
                    for &edge in &graph.transpose_edges
                        [graph.transpose_indptr[row]..graph.transpose_indptr[row + 1]]
                    {
                        let dest = graph.dst[edge] * batch;
                        for (value, &x) in out.iter_mut().zip(&input[dest..dest + batch]) {
                            *value += weights[edge] * x;
                        }
                    }
                })
        });
        output
    }
    pub fn edge_vjp(
        &self,
        graph: &Graph,
        input: &[f32],
        cotangent: &[f32],
        batch: usize,
        result: &mut [f32],
    ) {
        self.pool.install(|| {
            result.par_iter_mut().enumerate().for_each(|(edge, value)| {
                let source = graph.src[edge] * batch;
                let dest = graph.dst[edge] * batch;
                let mut sum = 0.0;
                for b in 0..batch {
                    sum += input[source + b] * cotangent[dest + b];
                }
                *value += sum;
            })
        });
    }
}
