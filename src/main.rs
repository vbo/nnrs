extern crate rand;

use std::fmt;
use rand::Rng;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug)]
struct Matrix {
    mem: Vec<f64>,
    cols: usize,
    rows: usize,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {}x{}:\n", self.rows, self.cols);
        for row in 0..self.rows {
            let row_start = row*self.cols;
            for col in 0..self.cols {
                write!(f, "{:8.4}", self.mem[row_start + col]);
            }
            write!(f, "\n");
        }
        
        return Ok(());
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        return Self {
            mem: Vec::with_capacity(cols * rows),
            rows: rows,
            cols: cols,
        };
    }

    pub fn init_with(mut self, val: f64) -> Self {
        for _ in 0..self.rows*self.cols {
            self.mem.push(val);
        }

        return self;
    }

    pub fn init_rand(mut self) -> Self {
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, 1.0);

        for _ in 0..self.rows*self.cols {
            self.mem.push(range.ind_sample(&mut rng));
        }

        return self;
    }

    pub fn dot_vec(&self, vec: &Vector, res: &mut Vector) {
        assert!(
            self.rows == res.rows && self.cols == vec.rows,
            "Dimentions invalid for product: \
             Matrix {}x{} * Vector {}x1 = Vector {}x1",
            self.rows, self.cols, vec.rows, res.rows);

        for row in 0..res.rows {
            let mat_row_start = row*self.cols;
            res.mem[row] = 0.0;
            for col in 0..self.cols {
                res.mem[row] += self.mem[mat_row_start + col] * vec.mem[col];
            }
        }
    }

    pub fn sub(&mut self, mat: &Matrix) {
        assert!(
            self.rows == mat.rows && self.cols == mat.cols,
            "Dimentions invalid for sub: \
             {}x{} != {}x{}",
            self.rows, self.cols, mat.rows, mat.cols);

        for i in 0..self.rows*self.cols {
            self.mem[i] -= mat.mem[i];
        }
    }

    pub fn add(&mut self, mat: &Matrix) {
        assert!(
            self.rows == mat.rows && self.cols == mat.cols,
            "Dimentions invalid for add: \
             {}x{} != {}x{}",
            self.rows, self.cols, mat.rows, mat.cols);

        for i in 0..self.rows*self.cols {
            self.mem[i] += mat.mem[i];
        }
    }

    pub fn apply<F>(&mut self, f: F)
                    where F: Fn(f64) -> f64 {
        for i in 0..self.rows*self.cols {
            self.mem[i] = f(self.mem[i]);
        }
    }

    pub fn transpose(&self, res: &mut Matrix) {
        assert!(
            self.rows == res.cols && self.cols == res.rows,
            "Dimentions invalid for transpose: {}x{}.T = {}x{}",
            self.rows, self.cols, res.rows, res.cols);

        for source_row in 0..self.rows {
            let res_col = source_row;
            let source_row_start = source_row*self.cols;
            for source_col in 0..self.cols {
                let res_row = source_col;
                let source_val = self.mem[source_row_start + source_col];
                res.mem[res_row*res.cols + res_col] = source_val;
            }
        }
    }
}


#[derive(Debug)]
struct Vector {
    mem: Vec<f64>,
    rows: usize,
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector {}x1:\n", self.rows);
        for row in 0..self.rows {
            write!(f, "{:8.4}\n", self.mem[row]);
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

    pub fn init_with(mut self, val: f64) -> Self {
        for _ in 0..self.rows {
            self.mem.push(val);
        }

        return self;
    }

    pub fn apply<F>(&mut self, f: F)
                    where F: Fn(f64) -> f64 {
        for row in 0..self.rows {
            self.mem[row] = f(self.mem[row]);
        }
    }

    pub fn copy_from(&mut self, v: &Vector) {
        assert!(self.rows == v.rows,
                "Invalid dimentions for copy_from: {}x1 vs {}x1",
                self.rows, v.rows);

        self.mem = v.mem.to_vec();
    }

    pub fn sub(&self, v: &Vector, res: &mut Vector) {
        assert!(self.rows == v.rows && self.rows == res.rows,
                "Invalid dimentions for sub: {}x1 - {}x1 = {}x1",
                self.rows, v.rows, res.rows);

        for row in 0..self.rows {
            res.mem[row] = self.mem[row] - v.mem[row];
        }
    }

    pub fn fill_rand(&mut self) {
        let mut rng = rand::thread_rng();
        let range = Range::new(0.0, 1.0);

        for i in 0..self.rows {
            self.mem[i] = range.ind_sample(&mut rng);
        }
    }
}

fn calc_output_weights_pd(
    output_error_grad_prefix: &Vector,
    l2_activations: &Vector,
    res: &mut Matrix) {
    assert!(output_error_grad_prefix.rows == res.rows && res.cols == l2_activations.rows,
            "Invalid dimentions for output weights PD: {}x1, {}x1, {}x{}",
            output_error_grad_prefix.rows, l2_activations.rows, res.rows, res.cols);

    for row in 0..res.rows {
        let row_start = row*res.cols;
        for col in 0..res.cols {
            res.mem[row_start + col] = output_error_grad_prefix.mem[row]
                                     * l2_activations.mem[col];
        }
    }
}

fn calc_output_error_grad_prefix(
    outputs: &Vector,
    error: &Vector,
    res: &mut Vector) {
    assert!(outputs.rows == error.rows && error.rows == res.rows,
            "Invalid dimentions for output error grad prefix PD: {}x1, {}x1, {}x1",
            outputs.rows, error.rows, res.rows);
    
    for row in 0..outputs.rows {
        res.mem[row] = 2.0*error.mem[row]*outputs.mem[row]*(1.0 - outputs.mem[row]);
    }
}

// Sizes
const N_INPUTS: usize = 2;
const N_L1: usize = 5;
const N_L2: usize = 5;
const N_OUTPUTS: usize = 2;

const BATCH_SIZE: usize = 2;

fn sigmoid(x: f64) -> f64 { 1.0 / ((-x).exp() + 1.0) }

// Weights matrix:
// Each row represents weights of edges between a given target node and all source nodes.
fn main() {
    let mut inputs = Vector::new(N_INPUTS).init_with(1.0);
    let mut l1_weights = Matrix::new(N_L1, N_INPUTS).init_rand();
    let mut l1_weights_pd = Matrix::new(N_L1, N_INPUTS).init_with(0.0);
    let mut l1_weights_t = Matrix::new(N_INPUTS, N_L1).init_rand();
    let mut l1_activations = Vector::new(N_L1).init_with(0.0);
    let mut l1_error = Vector::new(N_L1).init_with(0.0);
    let mut l1_error_grad_prefix = Vector::new(N_L1).init_with(0.0);
    let mut l2_weights = Matrix::new(N_L2, N_L1).init_rand();
    let mut l2_weights_pd = Matrix::new(N_L2, N_L1).init_with(0.0);
    let mut l2_weights_t = Matrix::new(N_L1, N_L2).init_rand();
    let mut l2_activations = Vector::new(N_L2).init_with(0.0);
    let mut l2_error = Vector::new(N_L2).init_with(0.0);
    let mut l2_error_grad_prefix = Vector::new(N_L2).init_with(0.0);
    let mut output_weights = Matrix::new(N_OUTPUTS, N_L2).init_rand();
    let mut output_weights_t = Matrix::new(N_L2, N_OUTPUTS).init_rand();
    let mut output_weights_pd = Matrix::new(N_OUTPUTS, N_L2).init_with(0.0);
    let mut output_error_grad_prefix = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);

    println!("before inputs:{}", inputs);
    //println!("before l1_weights:{}", l1_weights);
    //println!("before l1_activations:{}", l1_activations);
    //println!("before l2_weights:{}", l2_weights);
    //println!("before l2_activations:{}", l2_activations);
    //println!("before output_weights:{}", output_weights);
    //println!("before outputs:{}", outputs);

    for i in 0..1_000_000 {
        // Training data
        inputs.fill_rand();

        // Forward propagation
        l1_weights.dot_vec(&inputs, &mut l1_activations);
        l1_activations.apply(sigmoid);
        l2_weights.dot_vec(&l1_activations, &mut l2_activations);
        l2_activations.apply(sigmoid);
        output_weights.dot_vec(&l2_activations, &mut outputs);
        outputs.apply(sigmoid);
    
        // Error
        true_outputs.copy_from(&inputs);
        true_outputs.sub(&outputs, &mut error);
        
        // Backward propagation
        // Output layer
        calc_output_error_grad_prefix(&outputs, &error, &mut output_error_grad_prefix);
        calc_output_weights_pd(&output_error_grad_prefix,
                               &l2_activations, &mut output_weights_pd);
        output_weights.transpose(&mut output_weights_t);
        output_weights_t.dot_vec(&output_error_grad_prefix, &mut l2_error);
        
        // Hidden layer L2
        calc_output_error_grad_prefix(
            &l2_activations, &l2_error, &mut l2_error_grad_prefix);
        calc_output_weights_pd(&l2_error_grad_prefix,
                               &l1_activations, &mut l2_weights_pd);
        l2_weights.transpose(&mut l2_weights_t);
        l2_weights_t.dot_vec(&l2_error_grad_prefix, &mut l1_error);
        
        // Hidden layer L1
        calc_output_error_grad_prefix(
            &l1_activations, &l1_error, &mut l1_error_grad_prefix);
        calc_output_weights_pd(&l1_error_grad_prefix,
                               &inputs, &mut l1_weights_pd);
        //l1_weights.transpose(&mut l1_weights_t);
        //l1_weights_t.dot_vec(&l1_error_grad_prefix, &mut l1_error);

        // Weights adjustment
        // Output
        output_weights_pd.apply(|x| { x * 0.1 });
        output_weights.add(&output_weights_pd);
        // L2
        l2_weights_pd.apply(|x| { x * 0.1 });
        l2_weights.add(&l2_weights_pd);
        // L1
        l1_weights_pd.apply(|x| { x * 0.1 });
        l1_weights.add(&l1_weights_pd);

        if i % 100_000 == 0 {
            println!("error[{}] {:?}", i, error.mem);
        }
    }


    println!("after l1_weights:{}", l1_weights);
    println!("after l2_weights:{}", l2_weights);
    println!("after output_weights:{}", output_weights);
    println!("after error:{}", error);

    println!("after output_weights_pd:{}", output_weights_pd);
}
