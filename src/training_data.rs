extern crate byteorder;

use std::fs::File;
use std::io::Read;
use self::byteorder::{ByteOrder, BigEndian};

pub struct TrainingData {
    pub input_mem: Vec<f64>,
    pub label_mem: Vec<f64>,
    pub example_indices: Vec<usize>,
    pub examples_count: usize,
    pub input_size: usize,
    pub label_size: usize,
}

pub fn load_mnist() -> TrainingData {
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
