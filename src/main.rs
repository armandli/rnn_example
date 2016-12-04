extern crate rnn_example;
extern crate rulinalg;

use std::path::Path;

use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::Matrix;

use rnn_example::*;

extern crate rand;
use rand::Rng;

fn main() {
    let input = "./input.txt";
    println!("input = {}", input);

    let input_file = Path::new(input);
    let (data, enc, ctoe, etoc) = match read_paragraph(input_file) {
        Ok((a,b,c,d)) => (a,b,c,d),
        Err(e) => panic!("{}", e),
    };

    //this was good enough after 600 iterations
//    let rnn = RNNTextGen2::train(data, enc, etoc, ctoe, 100, 2000, 25, 1e-1);

    let rnn = LSTMTextGen::train(data, enc, etoc, ctoe, 100, 2000, 25, 1e-1, 0.99);
}
