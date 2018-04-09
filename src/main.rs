extern crate rand;
extern crate byteorder;

use std::fmt;
use std::fs::File;
use std::io::Read;

use rand::distributions::{IndependentSample, Range};
use byteorder::{ByteOrder, BigEndian};

#[derive(Debug)]
struct Matrix {
    mem: Vec<f64>,
    cols: usize,
    rows: usize,
}

struct TrainingData {
    input_mem: Vec<f64>,
    label_mem: Vec<f64>,
    example_indices: Vec<usize>,
    examples_count: usize,
    input_size: usize,
    label_size: usize,
}

fn load_mnist_training_data() -> TrainingData {
    const MNIST_IMAGES_FILE: &str = "data/train-images-idx3-ubyte.bin";
    const MNIST_LABELS_FILE: &str = "data/train-labels-idx1-ubyte.bin";
    let mut images_file = File::open(MNIST_IMAGES_FILE).unwrap();

    let mut header_buf = [0u8; 16];
    images_file.read_exact(&mut header_buf).unwrap();
    let images_count = BigEndian::read_i32(&header_buf[4..]) as usize;
    let rows = BigEndian::read_i32(&header_buf[8..]) as usize;
    let cols = BigEndian::read_i32(&header_buf[12..]) as usize;

    let input_size = rows*cols;
    let images_data_size = images_count*input_size;
    let mut images_data = Vec::<f64>::with_capacity(images_data_size);
    let mut read_buf = [0u8; 1024*1024];
    loop {
        let bytes_read = images_file.read(&mut read_buf).unwrap();
        if bytes_read == 0 {
            break;
        }

        for i in 0..bytes_read {
            images_data.push((read_buf[i] as f64)/255.0);
        }
    }

    let mut labels_file = File::open(MNIST_LABELS_FILE).unwrap();

    let mut header_buf = [0u8; 8];
    labels_file.read_exact(&mut header_buf).unwrap();
    let labels_count = BigEndian::read_i32(&header_buf[4..]) as usize;

    assert!(labels_count == images_count,
            "Invalid training data. Labels count != inputs count");

    let label_size = 10usize;
    let labels_data_size = labels_count*label_size;
    let mut labels_data = Vec::<f64>::with_capacity(labels_data_size);

    loop {
        let bytes_read = labels_file.read(&mut read_buf).unwrap();
        if bytes_read == 0 {
            break;
        }

        for i in 0..bytes_read {
            let mut label_found = false;
            for j in 0..10 {
                let mut data = 0.0f64;
                if read_buf[i] as usize == j {
                    data = 1.0f64;
                    label_found = true;
                }

                labels_data.push(data);
            }

            assert!(label_found, "Label not found byte: {}.", read_buf[i]);
        }
    }

    let res = TrainingData {
        input_mem: images_data,
        label_mem: labels_data,
        example_indices: (0usize..images_count).collect(),
        examples_count: images_count,
        input_size: input_size,
        label_size: label_size,
    };

    return res;
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
        let range = Range::new(-0.5, 0.5);

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

    #[allow(dead_code)]
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

    pub fn apply<F>(&mut self, f: &F)
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

const WRITE_ERR: &str = "Failed to write";

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix {}x{}:\n", self.rows, self.cols).expect(&WRITE_ERR);
        for row in 0..self.rows {
            let row_start = row*self.cols;
            for col in 0..self.cols {
                write!(f, "{:8.4}", self.mem[row_start + col]).expect(&WRITE_ERR);

            }
            write!(f, "\n").expect(&WRITE_ERR);

        }

        return Ok(());
    }
}


#[derive(Debug)]
struct Vector {
    mem: Vec<f64>,
    rows: usize,
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

    pub fn copy_from_slice(&mut self, s: &[f64]) {
        assert!(self.rows == s.len(),
                "Invalid dimentions for copy_from_slice: {}x1 vs {}x1",
                self.rows, s.len()); 
        for i in 0..self.rows {
            self.mem[i] = s[i];
        }
    }

    pub fn sub(&self, v: &Vector, res: &mut Vector) {
        assert!(self.rows == v.rows && self.rows == res.rows,
                "Invalid dimentions for sub: {}x1 - {}x1 = {}x1",
                self.rows, v.rows, res.rows);

        for row in 0..self.rows {
            res.mem[row] = self.mem[row] - v.mem[row];
        }
    }

    pub fn add_to_me(&mut self, v: &Vector) {
        assert!(self.rows == v.rows,
                "Invalid dimentions for add: {}x1 + {}x1",
                self.rows, v.rows);

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

    pub fn calc_length(&self) -> f64 {
        let mut res: f64 = 0.0;
        for i in 0..self.rows {
            res += self.mem[i]*self.mem[i];
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

fn calc_weights_pd(
    error_grad_prefix: &Vector,
    previous_activations: &Vector,
    res: &mut Matrix) {
    assert!(error_grad_prefix.rows == res.rows && res.cols == previous_activations.rows,
            "Invalid dimentions for output weights PD: {}x1, {}x1, {}x{}",
            error_grad_prefix.rows, previous_activations.rows, res.rows, res.cols);

    for row in 0..res.rows {
        let row_start = row*res.cols;
        for col in 0..res.cols {
            res.mem[row_start + col] = error_grad_prefix.mem[row]
                                     * previous_activations.mem[col];
        }
    }
}

fn calc_grad_prefix(
    activations: &Vector,
    error: &Vector,
    res: &mut Vector) {
    assert!(activations.rows == error.rows && error.rows == res.rows,
            "Invalid dimentions for output error grad prefix PD: {}x1, {}x1, {}x1",
            activations.rows, error.rows, res.rows);

    for row in 0..activations.rows {
        res.mem[row] = 2.0*error.mem[row]*activations.mem[row]*(1.0 - activations.mem[row]);
    }
}

fn render_training_example(label_data: &[f64], input_data: &[f64]) {
    for i in 0..10 {
        if label_data[i] > 0.0 {
            println!("Label: {}", i);
        }
    }

    for i in 0..28 {
        let row_start = i*28;
        for j in 0..28 {
            let v = input_data[row_start + j];
            if v > 0.7 {
                print!("1");
            } else {
                print!("0");
            }
        }
        print!("\n");
    }
}

// Sizes
const N_INPUTS: usize = 28*28;
const N_L1: usize = 16;
const N_L2: usize = 16;
const N_OUTPUTS: usize = 10;

const BATCH_SIZE: usize = 1000;
const LOG_EVERY_N: usize = 10_000;
const LEARNING_RATE: f64 = 0.1;
const NUM_EPOCHS: usize = 1000;

fn sigmoid(x: f64) -> f64 { 1.0 / ((-x).exp() + 1.0) }

// Weights matrix:
// Each row represents weights of edges between a given target node and all source nodes.
fn main() {
    let training_data = load_mnist_training_data();

    assert!(training_data.input_size == N_INPUTS, "Wrong inputs!");
    assert!(training_data.label_size == N_OUTPUTS, "Wrong outputs!");

    let mut inputs = Vector::new(N_INPUTS).init_with(1.0);

    let mut l1_weights = Matrix::new(N_L1, N_INPUTS).init_rand();
    let mut l1_weights_pd = Matrix::new(N_L1, N_INPUTS).init_with(0.0);
    let mut l1_weights_batch_pd = Matrix::new(N_L1, N_INPUTS).init_with(0.0);
    let mut l1_grad_prefix = Vector::new(N_L1).init_with(0.0);

    let mut l1_bias = 0.0f64;
    let mut l1_bias_batch_pd = 0.0f64;
    let mut l1_activations = Vector::new(N_L1).init_with(0.0);

    let mut l1_error = Vector::new(N_L1).init_with(0.0);

    let mut l2_weights = Matrix::new(N_L2, N_L1).init_rand();
    let mut l2_weights_pd = Matrix::new(N_L2, N_L1).init_with(0.0);
    let mut l2_weights_batch_pd = Matrix::new(N_L2, N_L1).init_with(0.0);
    let mut l2_weights_t = Matrix::new(N_L1, N_L2).init_with(0.0);
    let mut l2_grad_prefix = Vector::new(N_L2).init_with(0.0);

    let mut l2_bias = 0.0f64;
    let mut l2_bias_batch_pd = 0.0f64;
    let mut l2_activations = Vector::new(N_L2).init_with(0.0);

    let mut l2_error = Vector::new(N_L2).init_with(0.0);

    let mut output_weights = Matrix::new(N_OUTPUTS, N_L2).init_rand();
    let mut output_weights_t = Matrix::new(N_L2, N_OUTPUTS).init_with(0.0);
    let mut output_weights_pd = Matrix::new(N_OUTPUTS, N_L2).init_with(0.0);
    let mut output_weights_batch_pd = Matrix::new(N_OUTPUTS, N_L2).init_with(0.0);
    let mut output_grad_prefix = Vector::new(N_OUTPUTS).init_with(0.0);

    let mut output_bias = 0.0f64;
    let mut output_bias_batch_pd = 0.0f64;
    let mut outputs = Vector::new(N_OUTPUTS).init_with(0.0);

    let mut true_outputs = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut error = Vector::new(N_OUTPUTS).init_with(0.0);
    let mut avg_error: f64 = 0.0;

    let avg_by_batch = |x| { x * LEARNING_RATE / BATCH_SIZE as f64 };
    let set_0 = |_: f64| 0.0;


    let mut current_examples_cursor = 0usize;
    let mut hits = 0usize;
    let mut i = 1usize;
    while i < NUM_EPOCHS {
        // Training data
        let current_example_index = training_data.example_indices[current_examples_cursor];
        let input_data_offset = current_example_index*training_data.input_size;
        let input_data_end = input_data_offset + training_data.input_size;
        let input_data = &training_data.input_mem[input_data_offset..input_data_end];

        let label_data_offset = current_example_index*training_data.label_size;
        let label_data_end = label_data_offset + training_data.label_size;
        let label_data = &training_data.label_mem[label_data_offset..label_data_end];
        inputs.copy_from_slice(input_data);
        true_outputs.copy_from_slice(label_data);

        current_examples_cursor += 1;
        if current_examples_cursor >= training_data.examples_count {
            current_examples_cursor = 0;
            i += 1;
        }

        // Forward propagation
        l1_weights.dot_vec(&inputs, &mut l1_activations);
        l1_activations.apply(|x| {x + l1_bias});
        l1_activations.apply(sigmoid);

        l2_weights.dot_vec(&l1_activations, &mut l2_activations);
        l2_activations.apply(|x| {x + l2_bias});
        l2_activations.apply(sigmoid);

        output_weights.dot_vec(&l2_activations, &mut outputs);
        outputs.add_to_me(&output_grad_prefix);
        outputs.apply(|x| {x + output_bias});
        outputs.apply(sigmoid);

        // Error
        true_outputs.sub(&outputs, &mut error);
        avg_error += error.calc_length();
        let (max_i, max) = outputs.max_component();
        let (tmax_i, tmax) = true_outputs.max_component();
        if max_i == tmax_i {
            hits += 1;
        }

        // Backward propagation
        // Output layer
        calc_grad_prefix(&outputs, &error, &mut output_grad_prefix);
        calc_weights_pd(&output_grad_prefix, &l2_activations, &mut output_weights_pd);
        output_bias_batch_pd += output_grad_prefix.calc_sum();
        output_weights_batch_pd.add(&output_weights_pd);
        output_weights.transpose(&mut output_weights_t);
        output_weights_t.dot_vec(&output_grad_prefix, &mut l2_error);

        // Hidden layer L2
        calc_grad_prefix(&l2_activations, &l2_error, &mut l2_grad_prefix);
        calc_weights_pd(&l2_grad_prefix, &l1_activations, &mut l2_weights_pd);
        l2_bias_batch_pd += l2_grad_prefix.calc_sum();
        l2_weights_batch_pd.add(&l2_weights_pd);
        l2_weights.transpose(&mut l2_weights_t);
        l2_weights_t.dot_vec(&l2_grad_prefix, &mut l1_error);

        // Hidden layer L1
        calc_grad_prefix(&l1_activations, &l1_error, &mut l1_grad_prefix);
        calc_weights_pd(&l1_grad_prefix, &inputs, &mut l1_weights_pd);
        l1_bias_batch_pd += l1_grad_prefix.calc_sum();
        l1_weights_batch_pd.add(&l1_weights_pd);

        if (i*current_examples_cursor) % LOG_EVERY_N == 0 {
            println!("error over last {}: {:8.4}", LOG_EVERY_N, avg_error / LOG_EVERY_N as f64);
            println!("hits {}%", (hits as f64) * 100.0 / (LOG_EVERY_N as f64));
            avg_error = 0.0;
            hits = 0;
            if (i*current_examples_cursor) % (BATCH_SIZE * 100) == 0 {
                //println!("l1_weights:{}", l1_weights);
                //println!("l1_grad_prefix:{}", l1_grad_prefix);
                //println!("l2_weights:{}", l2_weights);
                //println!("l2_grad_prefix:{}", l2_grad_prefix);
                //println!("output_weights:{}", output_weights);
                //println!("output_grad_prefix:{}", output_grad_prefix);
                println!("output:{}", outputs);
            }
        }
        if (i*current_examples_cursor) % BATCH_SIZE == 0 {
            // Weights adjustment
            // Output
            output_bias += avg_by_batch(output_bias_batch_pd);
            output_bias_batch_pd = 0.0;

            output_weights_batch_pd.apply(&avg_by_batch);
            output_weights.add(&output_weights_batch_pd);
            output_weights_batch_pd.apply(&set_0);

            // L2
            l2_bias += avg_by_batch(l2_bias_batch_pd);
            l2_bias_batch_pd = 0.0;

            l2_weights_batch_pd.apply(&avg_by_batch);
            l2_weights.add(&l2_weights_batch_pd);
            l2_weights_batch_pd.apply(&set_0);

            // L1
            l1_bias += avg_by_batch(l1_bias_batch_pd);
            l1_bias_batch_pd = 0.0;

            l1_weights_batch_pd.apply(&avg_by_batch);
            l1_weights.add(&l1_weights_batch_pd);
            l1_weights_batch_pd.apply(&set_0);
        }
    }


    //println!("after l1_weights:{}", l1_weights);
    //println!("after l1_grad_prefix:{}", l1_grad_prefix);
    //println!("after l2_weights:{}", l2_weights);
    //println!("after l2_grad_prefix:{}", l2_grad_prefix);
    //println!("after output_weights:{}", output_weights);
    //println!("after output_grad_prefix:{}", output_grad_prefix);
    //println!("after error:{}", error);
}
