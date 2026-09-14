//! Validated destination CSR plus a transpose pointing to canonical edge IDs.
#[derive(Clone)]
pub struct Graph {
    pub indptr: Vec<usize>,
    pub src: Vec<usize>,
    pub dst: Vec<usize>,
    pub transpose_indptr: Vec<usize>,
    pub transpose_edges: Vec<usize>,
    /// Original destinations, contiguous in transpose row order.
    pub transpose_src: Vec<usize>,
    pub type_id: Vec<usize>,
    pub sign: Vec<f32>,
    pub types: usize,
}

impl Graph {
    pub fn new(indptr: &[i32], src: &[i32], type_id: &[i32], sign: &[f32]) -> Result<Self, String> {
        let n = type_id.len();
        if n == 0
            || indptr.len() != n + 1
            || sign.len() != n
            || indptr[0] != 0
            || indptr[n] as usize != src.len()
            || indptr.windows(2).any(|x| x[0] > x[1])
            || src.iter().any(|&x| x < 0 || x as usize >= n)
            || type_id.iter().any(|&x| x < 0)
            || sign.iter().any(|&x| x != 1.0 && x != -1.0)
        {
            return Err("Invalid graph CSR, type or sign arrays".into());
        }
        let indptr: Vec<_> = indptr.iter().map(|&x| x as usize).collect();
        let src: Vec<_> = src.iter().map(|&x| x as usize).collect();
        let mut dst = vec![0; src.len()];
        for row in 0..n {
            if src[indptr[row]..indptr[row + 1]]
                .windows(2)
                .any(|x| x[0] >= x[1])
            {
                return Err("Edges must be unique and sorted by destination then source".into());
            }
            dst[indptr[row]..indptr[row + 1]].fill(row);
        }
        let mut transpose_indptr = vec![0; n + 1];
        for &source in &src {
            transpose_indptr[source + 1] += 1;
        }
        for i in 0..n {
            transpose_indptr[i + 1] += transpose_indptr[i];
        }
        let mut cursor = transpose_indptr.clone();
        let mut transpose_edges = vec![0; src.len()];
        let mut transpose_src = vec![0; src.len()];
        for (edge, &source) in src.iter().enumerate() {
            transpose_edges[cursor[source]] = edge;
            transpose_src[cursor[source]] = dst[edge];
            cursor[source] += 1;
        }
        Ok(Self {
            indptr,
            src,
            dst,
            transpose_indptr,
            transpose_edges,
            transpose_src,
            type_id: type_id.iter().map(|&x| x as usize).collect(),
            sign: sign.to_vec(),
            types: 1 + *type_id.iter().max().unwrap() as usize,
        })
    }
    pub fn neurons(&self) -> usize {
        self.type_id.len()
    }
    pub fn edges(&self) -> usize {
        self.src.len()
    }
}
