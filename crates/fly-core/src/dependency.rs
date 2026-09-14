//! Exact finite-horizon dependencies of the readout, including each leak self-edge.
use crate::Graph;
use std::sync::Arc;

pub(crate) struct ReadoutPlan {
    /// Required voltage rows after passes 1..=K. Initial voltages are constant.
    pub required: Vec<Arc<[bool]>>,
}

impl ReadoutPlan {
    pub fn new(graph: &Graph, outputs: &[i32], steps: usize) -> Result<Self, String> {
        if steps == 0 || steps > 1024 || outputs.len() != graph.neurons() {
            return Err("Invalid readout dependency shape or step count".into());
        }
        let final_rows: Arc<[bool]> = outputs.iter().map(|&group| group >= 0).collect();
        let mut reverse = vec![final_rows];
        while reverse.len() < steps {
            let previous = reverse.last().unwrap();
            // v_i[t+1] also depends on v_i[t], even without an explicit i->i edge.
            let mut rows = previous.to_vec();
            for (destination, &required) in previous.iter().enumerate() {
                if required {
                    for &source in &graph.src[graph.indptr[destination]..graph.indptr[destination + 1]] {
                        rows[source] = true;
                    }
                }
            }
            if rows.as_slice() == previous.as_ref() {
                // Strongly recurrent graphs usually reach a fixed point quickly.
                reverse.resize(steps, previous.clone());
                break;
            }
            reverse.push(rows.into());
        }
        reverse.reverse();
        Ok(Self { required: reverse })
    }

    pub fn counts(&self, graph: &Graph) -> Vec<(usize, usize)> {
        self.required.iter().map(|rows| {
            let mut neurons = 0;
            let mut edges = 0;
            for (node, &required) in rows.iter().enumerate() {
                if required {
                    neurons += 1;
                    edges += graph.indptr[node + 1] - graph.indptr[node];
                }
            }
            (neurons, edges)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readout_cone_includes_leak_cycles_and_excludes_disconnected_rows() {
        // 0->1->2->3 with a 2->1 cycle; node 4 has an unrelated self-loop.
        let graph = Graph::new(&[0, 0, 2, 3, 4, 5], &[0, 2, 1, 2, 4],
            &[0; 5], &[1.0; 5]).unwrap();
        let plan = ReadoutPlan::new(&graph, &[-1, -1, -1, 0, -1], 5).unwrap();
        let expected = [
            [true, true, true, true, false],
            [true, true, true, true, false],
            [false, true, true, true, false],
            [false, false, true, true, false],
            [false, false, false, true, false],
        ];
        for (actual, expected) in plan.required.iter().zip(expected) {
            assert_eq!(actual.as_ref(), &expected);
        }
        assert!(Arc::ptr_eq(&plan.required[0], &plan.required[1]));
        assert_eq!(plan.counts(&graph), [(4, 4), (4, 4), (3, 4), (2, 2), (1, 1)]);
        assert!(ReadoutPlan::new(&graph, &[-1; 5], 0).is_err());
        assert!(ReadoutPlan::new(&graph, &[-1; 5], 1025).is_err());
        assert!(ReadoutPlan::new(&graph, &[-1; 4], 2).is_err());
    }
}
