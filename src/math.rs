extern crate rand;

use rand::Rng;
use rand::distributions::{IndependentSample, Range};
use std::fmt;

#[derive(Debug)]
pub struct Matrix {
    pub mem: Vec<f64>,
    pub cols: usize,
    pub rows: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        return Self {
            mem: Vec::with_capacity(cols * rows),
            rows: rows,
            cols: cols,
        };
    }

    pub fn new_same_dim(m: &Matrix) -> Self {
        let cols = m.cols;
        let rows = m.rows;
        return Matrix::new(rows, cols);
    }

    pub fn init_with(mut self, val: f64) -> Self {
        for _ in 0..self.rows * self.cols {
            self.mem.push(val);
        }

        return self;
    }

    pub fn init_rand(mut self) -> Self {
        let mut rng = rand::thread_rng();
        let range = Range::new(-0.5, 0.5);

        for _ in 0..self.rows * self.cols {
            self.mem.push(range.ind_sample(&mut rng));
        }

        return self;
    }

    pub fn dot_vec(&self, vec: &Vector, res: &mut Vector) {
        assert!(
            self.rows == res.rows && self.cols == vec.rows,
            "Dimensions invalid for product: \
             Matrix {}x{} * Vector {}x1 = Vector {}x1",
            self.rows,
            self.cols,
            vec.rows,
            res.rows
        );

        for row in 0..res.rows {
            let mat_row_start = row * self.cols;
            res.mem[row] = 0.0;
            for col in 0..self.cols {
                res.mem[row] += self.mem[mat_row_start + col] * vec.mem[col];
            }
        }
    }

    pub fn add_dot_vec(&self, vec: &Vector, res: &mut Vector) {
        assert!(
            self.rows == res.rows && self.cols == vec.rows,
            "Dimensions invalid for add product: \
             Matrix {}x{} * Vector {}x1 = Vector {}x1",
            self.rows,
            self.cols,
            vec.rows,
            res.rows
        );

        for row in 0..res.rows {
            let mat_row_start = row * self.cols;
            for col in 0..self.cols {
                res.mem[row] += self.mem[mat_row_start + col] * vec.mem[col];
            }
        }
    }

    #[allow(dead_code)]
    pub fn sub(&mut self, mat: &Matrix) {
        assert!(
            self.rows == mat.rows && self.cols == mat.cols,
            "Dimensions invalid for sub: \
             {}x{} != {}x{}",
            self.rows,
            self.cols,
            mat.rows,
            mat.cols
        );

        for i in 0..self.rows * self.cols {
            self.mem[i] -= mat.mem[i];
        }
    }

    pub fn add(&mut self, mat: &Matrix) {
        assert!(
            self.rows == mat.rows && self.cols == mat.cols,
            "Dimensions invalid for add: \
             {}x{} != {}x{}",
            self.rows,
            self.cols,
            mat.rows,
            mat.cols
        );

        for i in 0..self.rows * self.cols {
            self.mem[i] += mat.mem[i];
        }
    }

    pub fn apply<F>(&mut self, f: &F)
    where
        F: Fn(f64) -> f64,
    {
        for i in 0..self.rows * self.cols {
            self.mem[i] = f(self.mem[i]);
        }
    }

    pub fn transpose(&self, res: &mut Matrix) {
        assert!(
            self.rows == res.cols && self.cols == res.rows,
            "Dimensions invalid for transpose: {}x{}.T = {}x{}",
            self.rows,
            self.cols,
            res.rows,
            res.cols
        );

        for source_row in 0..self.rows {
            let res_col = source_row;
            let source_row_start = source_row * self.cols;
            for source_col in 0..self.cols {
                let res_row = source_col;
                let source_val = self.mem[source_row_start + source_col];
                res.mem[res_row * res.cols + res_col] = source_val;
            }
        }
    }
}

const WRITE_ERR: &str = "Failed to write";

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {}x{}:\n", self.rows, self.cols).expect(&WRITE_ERR);
        for row in 0..self.rows {
            let row_start = row * self.cols;
            for col in 0..self.cols {
                write!(f, "{:8.4}", self.mem[row_start + col]).expect(&WRITE_ERR);
            }
            write!(f, "\n").expect(&WRITE_ERR);
        }

        return Ok(());
    }
}

#[derive(Debug)]
pub struct Vector {
    pub mem: Vec<f64>,
    pub rows: usize,
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector {}x1:\n", self.rows).expect(&WRITE_ERR);
        for row in 0..self.rows {
            write!(f, "{:8.4}\n", self.mem[row]).expect(&WRITE_ERR);
        }

        return Ok(());
    }
}

impl Vector {
    pub fn new(rows: usize) -> Self {
        return Self {
            mem: Vec::with_capacity(rows),
            rows: rows,
        };
    }

    pub fn empty() -> Self {
        return Self {
            mem: Vec::new(),
            rows: 0,
        };
    }

    pub fn init_with(mut self, val: f64) -> Self {
        for _ in 0..self.rows {
            self.mem.push(val);
        }

        return self;
    }

    pub fn init_rand(mut self) -> Self {
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, 1.0);

        for _ in 0..self.rows {
            self.mem.push(range.ind_sample(&mut rng));
        }
        return self;
    }

    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for row in 0..self.rows {
            self.mem[row] = f(self.mem[row]);
        }
    }

    pub fn copy_from(&mut self, v: &Vector) {
        assert!(
            self.rows == v.rows,
            "Invalid dimensions for copy_from: {}x1 vs {}x1",
            self.rows,
            v.rows
        );
        self.mem = v.mem.to_vec();
    }

    pub fn copy_from_slice(&mut self, s: &[f64]) {
        assert!(
            self.rows == s.len(),
            "Invalid dimensions for copy_from_slice: {}x1 vs {}x1",
            self.rows,
            s.len()
        );
        for i in 0..self.rows {
            self.mem[i] = s[i];
        }
    }

    pub fn sub(&self, v: &Vector, res: &mut Vector) {
        assert!(
            self.rows == v.rows && self.rows == res.rows,
            "Invalid dimensions for sub: {}x1 - {}x1 = {}x1",
            self.rows,
            v.rows,
            res.rows
        );

        for row in 0..self.rows {
            res.mem[row] = self.mem[row] - v.mem[row];
        }
    }

    pub fn add_to_me(&mut self, v: &Vector) {
        assert!(
            self.rows == v.rows,
            "Invalid dimensions for add: {}x1 + {}x1",
            self.rows,
            v.rows
        );

        for row in 0..self.rows {
            self.mem[row] = self.mem[row] + v.mem[row];
        }
    }

    pub fn fill_rand(&mut self) {
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, 1.0);

        for i in 0..self.rows {
            self.mem[i] = range.ind_sample(&mut rng);
        }
    }

    pub fn fill_with(&mut self, v: f64) {
        for i in 0..self.rows {
            self.mem[i] = v;
        }
    }

    pub fn calc_length(&self) -> f64 {
        let mut res: f64 = 0.0;
        for i in 0..self.rows {
            res += self.mem[i] * self.mem[i];
        }
        return res.sqrt();
    }

    pub fn calc_sum(&self) -> f64 {
        let mut res: f64 = 0.0;
        for i in 0..self.rows {
            res += self.mem[i];
        }
        return res;
    }

    pub fn max_component(&self) -> (usize, f64) {
        let mut max = self.mem[0];
        let mut max_i = 0;

        for i in 1..self.rows {
            let val = self.mem[i];
            if val > max {
                max = val;
                max_i = i;
            }
        }

        return (max_i, max);
    }
}
