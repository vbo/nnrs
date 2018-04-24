extern crate byteorder;

use std::fs::File;
use std::io::Read;
use self::byteorder::{BigEndian, ByteOrder};

pub struct Dataset {
    pub input_mem: Vec<f64>,
    pub label_mem: Vec<f64>,
    pub example_indices: Vec<usize>,
    pub examples_count: usize,
    pub input_size: usize,
    pub label_size: usize,
}

impl Dataset {
    pub fn slices_for_cursor(&self, cursor: usize) -> (&[f64], &[f64]) {
        let current_example_index = self.example_indices[cursor];

        let input_data_offset = current_example_index * self.input_size;
        let input_data_end = input_data_offset + self.input_size;
        let input_data = &self.input_mem[input_data_offset..input_data_end];

        let label_data_offset = current_example_index * self.label_size;
        let label_data_end = label_data_offset + self.label_size;
        let label_data = &self.label_mem[label_data_offset..label_data_end];

        return (input_data, label_data);
    }
}

pub fn load_mnist_testing() -> Dataset {
    load_mnist(
        "data/t10k-images-idx3-ubyte.bin",
        "data/t10k-labels-idx1-ubyte.bin",
    )
}

pub fn load_mnist_training() -> Dataset {
    load_mnist(
        "data/train-images-idx3-ubyte.bin",
        "data/train-labels-idx1-ubyte.bin",
    )
}

pub fn load_mnist(images_file_path: &str, labels_file_path: &str) -> Dataset {
    let mut images_file = File::open(images_file_path).unwrap();

    let mut header_buf = [0u8; 16];
    images_file.read_exact(&mut header_buf).unwrap();
    let images_count = BigEndian::read_i32(&header_buf[4..]) as usize;
    let rows = BigEndian::read_i32(&header_buf[8..]) as usize;
    let cols = BigEndian::read_i32(&header_buf[12..]) as usize;

    let input_size = rows * cols;
    let images_data_size = images_count * input_size;
    let mut images_data = Vec::<f64>::with_capacity(images_data_size);
    let mut read_buf = [0u8; 1024 * 1024];
    loop {
        let bytes_read = images_file.read(&mut read_buf).unwrap();
        if bytes_read == 0 {
            break;
        }

        for i in 0..bytes_read {
            images_data.push((read_buf[i] as f64) / 255.0);
        }
    }

    let mut labels_file = File::open(labels_file_path).unwrap();

    let mut header_buf = [0u8; 8];
    labels_file.read_exact(&mut header_buf).unwrap();
    let labels_count = BigEndian::read_i32(&header_buf[4..]) as usize;

    assert!(
        labels_count == images_count,
        "Invalid training data. Labels count != inputs count"
    );

    let label_size = 10usize;
    let labels_data_size = labels_count * label_size;
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

    let res = Dataset {
        input_mem: images_data,
        label_mem: labels_data,
        example_indices: (0usize..images_count).collect(),
        examples_count: images_count,
        input_size: input_size,
        label_size: label_size,
    };

    return res;
}
