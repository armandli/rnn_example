extern crate rand;
extern crate rulinalg;

use std::cmp::min;
use std::fs::File;
use std::io::{Read, Error};
use std::path::Path;
use std::collections::HashMap;

use rand::Rng;
use rand::ThreadRng;

use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::Matrix;

pub fn read_paragraph(path: &Path) -> Result<(Vec<usize>, usize, HashMap<char, usize>, HashMap<usize, char>), Error> {
    let mut f = try!(File::open(path));
    let mut buf = String::new();
    f.read_to_string(&mut buf).expect("cannot read input file to string");
    let data = String::from(buf.trim());

    println!("{}", data);
    println!("len: {}", data.len());

    let mut emax = 0;
    let mut etoc: HashMap<usize, char> = HashMap::new();
    let mut ctoe: HashMap<char, usize> = HashMap::new();
    let mut encoded: Vec<usize> = Vec::new();

    for c in data.chars() {
        if ctoe.contains_key(&c) {
            let k = ctoe.get(&c).expect("c does not exist in ctoe after contains_key test");
            encoded.push(*k);
        } else {
            ctoe.insert(c, emax);
            etoc.insert(emax, c);
            encoded.push(emax);
            emax += 1;
        }
    }

    Ok((encoded, emax, ctoe, etoc))
}

pub struct RNNTextGen {
    Wxh: Matrix<f64>,
    Whh: Matrix<f64>,
    Who: Matrix<f64>,
    Bh: Matrix<f64>,
    Bo: Matrix<f64>,
    ctoe: HashMap<char, usize>,
    etoc: HashMap<usize, char>,
}

fn rmtx(r: usize, c: usize) -> Matrix<f64> {
    let mut rng = rand::thread_rng();
    let f = |_, _| {
        //why is the random matrix initial values has to be so small!! e.g. try 1. and you'll see
        //it's not converging fast enough
        let mut k = rng.gen::<f64>() * 0.1;
        if rng.gen::<u32>() % 100 < 50 {
            k * -1.
        } else {
            k
        }
    };
    Matrix::from_fn(r, c, f)
}

fn create_input(input: &Vec<usize>, enc: usize) -> Vec<Matrix<f64>> {
    input.iter().fold(Vec::new(), |mut v, i| {
        let mut iv = vec![0.; enc];
        iv[*i] = 1.;
        let mtx = Matrix::new(1, enc, iv);
        v.push(mtx);
        v
    })
}

fn softmax(m: &mut Matrix<f64>) {
    for v in m.mut_data().iter_mut() {
        *v = v.exp();
    }
    for i in 0..m.rows() {
        let row = m.get_row_mut(i).expect("invalid row index on m");
        let sum = row.iter().fold(0., |s, v| {
            s + v
        });
        for v in row {
            let val = *v;
            *v = val / sum;
        }
    }
}

fn mtx_to_c(mtx: &Matrix<f64>, etoc: &HashMap<usize, char>) -> char {
    let mut idx = 0;
    let r = mtx.get_row(0).expect("row index 0 out of bound");
    for i in 0..r.len() {
        if r[i] > r[idx] {
            idx = i;
        }
    }
    match etoc.get(&idx) {
        Some(c) => *c,
        None => panic!("enum value not encoded into etoc map"),
    }
}

fn clip(mtx: &mut Matrix<f64>) {
    for v in mtx.mut_data().iter_mut() {
        if *v > 5. {
            *v = 5.;
        } else if *v < -5. {
            *v = -5.;
        }
    }
}

fn adagrad(w: &mut Matrix<f64>, dw: &mut Matrix<f64>, mw: &mut Matrix<f64>, step: f64){
    for (mv, dv) in mw.mut_data().iter_mut().zip(dw.data().iter()) {
        *mv += dv * dv;
    }
    for (dv, mv) in dw.mut_data().iter_mut().zip(mw.data().iter()) {
        let val = *dv;
        let mut k = *mv;
        if ! k.is_normal() {
            k = 1e-8;
        }  else {
            k += 1e-8;
        }
        *dv = -1. * step * val / k.sqrt();
    }
    for (wv, dv) in w.mut_data().iter_mut().zip(dw.data().iter()) {
        *wv += *dv;
    }
}

fn rmsprop(w: &mut Matrix<f64>, dw: &mut Matrix<f64>, mw: &mut Matrix<f64>, step: f64, decay: f64) {
    for (mv, dv) in mw.mut_data().iter_mut().zip(dw.data().iter()) {
        let v = *mv;
        *mv = decay * v + (1. - decay) * dv * dv;
    }
    for (dv, mv) in dw.mut_data().iter_mut().zip(mw.data().iter()) {
        let val = *dv;
        let mut k = *mv;
        if ! k.is_normal() {
            k = 1e-8;
        } else {
            k += 1e-8;
        }
        *dv = -1. * step * val / k.sqrt();
    }
    for (wv, dv) in w.mut_data().iter_mut().zip(dw.data().iter()) {
        *wv += *dv;
    }
}


impl RNNTextGen {
    fn sample(Wxh: &Matrix<f64>, Whh: &Matrix<f64>, Who: &Matrix<f64>, Bh: &Matrix<f64>, Bo: &Matrix<f64>, etoc: &HashMap<usize, char>, hdim: usize, seed: Matrix<f64>, length: usize) -> String {
        let mut gen: Vec<Matrix<f64>> = Vec::new();
        gen.push(seed);
    
        let mut H0: Matrix<f64> = Matrix::zeros(1, hdim);
    
        for i in 0..length {
            let mut H = &gen[i] * Wxh + H0 * Whh + Bh;
            for v in H.mut_data().iter_mut() {
                *v = v.tanh();
            }
            let mut O = &H * Who + Bo;
            softmax(&mut O);
            gen.push(O);
            H0 = H;
        }
    
        let mut ret = String::new();
        for g in gen.iter() {
            let c = mtx_to_c(g, etoc);
            ret.push(c);
        }
        ret
    }

    pub fn train(input: Vec<usize>, enc: usize, etoc: HashMap<usize, char>, ctoe: HashMap<char, usize>, hdim: usize, iter: usize, slen: usize, step: f64) -> RNNTextGen {
        // initialization
        let mut Wxh: Matrix<f64> = rmtx(enc, hdim);
        let mut Whh: Matrix<f64> = rmtx(hdim, hdim);
        let mut Who: Matrix<f64> = rmtx(hdim, enc);
        let mut Bh: Matrix<f64> = Matrix::zeros(1, hdim);
        let mut Bo: Matrix<f64> = Matrix::zeros(1, enc);

        let mut mWxh: Matrix<f64> = Matrix::zeros(enc, hdim);
        let mut mWhh: Matrix<f64> = Matrix::zeros(hdim, hdim);
        let mut mWho: Matrix<f64> = Matrix::zeros(hdim, enc);
        let mut mBh: Matrix<f64> = Matrix::zeros(1, hdim);
        let mut mBo: Matrix<f64> = Matrix::zeros(1, enc);

        let data = create_input(&input, enc);

        let mut smooth_loss = -1. * (1. / enc as f64).log(std::f64::consts::E) * slen as f64;

        for k in 0..iter {
            let mut ss = 0;
            let mut Hprev: Matrix<f64> = Matrix::zeros(1, hdim);

            while ss < data.len() - 1 {
                let mut loss = 0.;
                let mut Os: Vec<Matrix<f64>> = Vec::new();
                let mut Hs: Vec<Matrix<f64>> = Vec::new();
                Hs.push(Hprev.clone());

                //forward pass
                for i in ss..min(ss + slen, data.len() - 1) {
                    let mut Hn = &data[i] * &Wxh + &Hs[i - ss] * &Whh + &Bh;
                    for v in Hn.mut_data().iter_mut() {
                        *v = v.tanh();
                    }
                    let mut O = &Hn * &Who + &Bo;
                    softmax(&mut O);
                    Os.push(O);
                    Hs.push(Hn);
                }

                //compute loss
                for i in ss..min(ss + slen, data.len() - 1) {
                    let or = Os[i - ss].get_row(0).expect("invalid row index 0 for computed output");
                    let idx = input[i+1];
                    loss += or[idx].log(std::f64::consts::E) * -1.;
                }
                smooth_loss = smooth_loss * 0.999 + loss * 0.001;

                //backpropagation
                let mut dWxh: Matrix<f64> = Matrix::zeros(enc, hdim);
                let mut dWhh: Matrix<f64> = Matrix::zeros(hdim, hdim);
                let mut dWho: Matrix<f64> = Matrix::zeros(hdim, enc);
                let mut dBh: Matrix<f64> = Matrix::zeros(1, hdim);
                let mut dBo: Matrix<f64> = Matrix::zeros(1, enc);
                let mut Hn: Matrix<f64> = Matrix::zeros(1, hdim);
                let mut j: i64 = min(ss + slen - 1, data.len() - 2) as i64;

                while j >= ss as i64 {
                    let o = &Os[j as usize - ss];
                    let y = &data[j as usize + 1];
                    let h = &Hs[j as usize - ss + 1];
                    let ht1 = &Hs[j as usize - ss];

                    let dY = o - y;

                    dWho += h.transpose() * &dY;
                    dBo += &dY;

                    let mut dH = &dY * Who.transpose() + &Hn;
                    for (dh, h) in dH.mut_data().iter_mut().zip(h.data().iter()) {
                        let k = (1. - h * h) * *dh;
                        *dh = k;
                    }
                    dWxh += data[j as usize].transpose() * &dH;
                    dWhh += ht1.transpose() * &dH;
                    dBh += &dH;
                    Hn = &dH * Whh.transpose();

                    j -= 1;
                }

                //regularization
                //tried it without the clipping, it works, is this step really necessary
                //in the long run?
                clip(&mut dWho);
                clip(&mut dWhh);
                clip(&mut dWxh);
                clip(&mut dBh);
                clip(&mut dBo);

                //adagrad parameter update, which is very important to be used
                //NOTE: adagrad is absolutely essential to make the ml work, why?? why is
                //simple update just doesn't work
                adagrad(&mut Who, &mut dWho, &mut mWho, step);
                adagrad(&mut Whh, &mut dWhh, &mut mWhh, step);
                adagrad(&mut Wxh, &mut dWxh, &mut mWxh, step);
                adagrad(&mut Bh, &mut dBh, &mut mBh, step);
                adagrad(&mut Bo, &mut dBo, &mut mBo, step);

                Hprev = Hs[Hs.len() - 1].clone();
                ss += slen;
            }

            if k % 100 == 0 {
                println!("{} = {}", k, smooth_loss);
                let seed = data[0].clone();
                println!("{}", Self::sample(&Wxh, &Whh, &Who, &Bh, &Bo, &etoc, hdim, seed, input.len()));
            }
        }

        RNNTextGen{ Wxh: Wxh, Whh: Whh, Who: Who, Bh: Bh, Bo: Bo, ctoe: ctoe, etoc: etoc }
    }
}

//removed the base vectors, this cause problems, result is not as good
pub struct RNNTextGen2 {
    Wxh: Matrix<f64>,
    Whh: Matrix<f64>,
    Who: Matrix<f64>,
    ctoe: HashMap<char, usize>,
    etoc: HashMap<usize, char>,
}

impl RNNTextGen2 {
    fn sample(Wxh: &Matrix<f64>, Whh: &Matrix<f64>, Who: &Matrix<f64>, etoc: &HashMap<usize, char>, hdim: usize, seed: Matrix<f64>, length: usize) -> String {
        let mut gen: Vec<Matrix<f64>> = Vec::new();
        gen.push(seed);
    
        let mut H0: Matrix<f64> = Matrix::zeros(1, hdim);
    
        for i in 0..length {
            let mut H = &gen[i] * Wxh + H0 * Whh;
            for v in H.mut_data().iter_mut() {
                *v = v.tanh();
            }
            let mut O = &H * Who;
            softmax(&mut O);
            gen.push(O);
            H0 = H;
        }
    
        let mut ret = String::new();
        for g in gen.iter() {
            let c = mtx_to_c(g, etoc);
            ret.push(c);
        }
        ret
    }

    pub fn train(input: Vec<usize>, enc: usize, etoc: HashMap<usize, char>, ctoe: HashMap<char, usize>, hdim: usize, iter: usize, slen: usize, step: f64) -> RNNTextGen2 {
        // initialization
        let mut Wxh: Matrix<f64> = rmtx(enc, hdim);
        let mut Whh: Matrix<f64> = rmtx(hdim, hdim);
        let mut Who: Matrix<f64> = rmtx(hdim, enc);

        let mut mWxh: Matrix<f64> = Matrix::zeros(enc, hdim);
        let mut mWhh: Matrix<f64> = Matrix::zeros(hdim, hdim);
        let mut mWho: Matrix<f64> = Matrix::zeros(hdim, enc);

        let data = create_input(&input, enc);

        let mut smooth_loss = -1. * (1. / enc as f64).log(std::f64::consts::E) * slen as f64;

        for k in 0..iter {
            let mut ss = 0;
            let mut Hprev: Matrix<f64> = Matrix::zeros(1, hdim);

            while ss < data.len() - 1 {
                let mut loss = 0.;
                let mut Os: Vec<Matrix<f64>> = Vec::new();
                let mut Hs: Vec<Matrix<f64>> = Vec::new();
                Hs.push(Hprev.clone());

                //forward path
                for i in ss..min(ss + slen, data.len() - 1) {
                    let mut Hn = &data[i] * &Wxh + &Hs[i - ss] * &Whh;
                    for v in Hn.mut_data().iter_mut() {
                        *v = v.tanh();
                    }
                    let mut O = &Hn * &Who;
                    softmax(&mut O);
                    Os.push(O);
                    Hs.push(Hn);
                }

                //compute loss
                for i in ss..min(ss + slen, data.len() - 1) {
                    let or = Os[i - ss].get_row(0).expect("invalid row index 0 for computed output");
                    let idx = input[i+1];
                    loss += or[idx].log(std::f64::consts::E) * -1.;
                }
                smooth_loss = smooth_loss * 0.999 + loss * 0.001;

                //backpropagation
                let mut dWxh: Matrix<f64> = Matrix::zeros(enc, hdim);
                let mut dWhh: Matrix<f64> = Matrix::zeros(hdim, hdim);
                let mut dWho: Matrix<f64> = Matrix::zeros(hdim, enc);
                let mut Hn: Matrix<f64> = Matrix::zeros(1, hdim);
                let mut j: i64 = min(ss + slen - 1, data.len() - 2) as i64;

                while j >= ss as i64 {
                    let o = &Os[j as usize - ss];
                    let y = &data[j as usize + 1];
                    let h = &Hs[j as usize - ss + 1];
                    let ht1 = &Hs[j as usize - ss];

                    let dY = o - y;

                    dWho += h.transpose() * &dY;

                    let mut dH = &dY * Who.transpose() + &Hn;
                    for (dh, h) in dH.mut_data().iter_mut().zip(h.data().iter()) {
                        let k = (1. - h * h) * *dh;
                        *dh = k;
                    }
                    dWxh += data[j as usize].transpose() * &dH;
                    dWhh += ht1.transpose() * &dH;
                    Hn = &dH * Whh.transpose();

                    j -= 1;
                }

                //regularization
                //tried it without the clipping, it works, is this step really necessary
                //in the long run?
                clip(&mut dWho);
                clip(&mut dWhh);
                clip(&mut dWxh);

                //adagrad parameter update, which is very important to be used
                //NOTE: adagrad is absolutely essential to make the ml work, why?? why is
                //simple update just doesn't work
                adagrad(&mut Who, &mut dWho, &mut mWho, step);
                adagrad(&mut Whh, &mut dWhh, &mut mWhh, step);
                adagrad(&mut Wxh, &mut dWxh, &mut mWxh, step);

                Hprev = Hs[Hs.len() - 1].clone();
                ss += slen;
            }

            if k % 100 == 0 {
                println!("{} = {}", k, smooth_loss);
                let seed = data[0].clone();
                println!("{}", Self::sample(&Wxh, &Whh, &Who, &etoc, hdim, seed, input.len()));
            }
        }

        RNNTextGen2{ Wxh: Wxh, Whh: Whh, Who: Who, ctoe: ctoe, etoc: etoc }

    }
}

pub struct LSTMTextGen {
    Wc: Matrix<f64>,
    Uc: Matrix<f64>,
    Wi: Matrix<f64>,
    Ui: Matrix<f64>,
    Wf: Matrix<f64>,
    Uf: Matrix<f64>,
    Wo: Matrix<f64>,
    ctoe: HashMap<char, usize>,
    etoc: HashMap<usize, char>,
}

impl LSTMTextGen {
    fn sample(Wc: &Matrix<f64>, Uc: &Matrix<f64>, Wi: &Matrix<f64>, Ui: &Matrix<f64>, Wf: &Matrix<f64>, Uf: &Matrix<f64>, Wo: &Matrix<f64>, etoc: &HashMap<usize, char>, hdim: usize, seed: Matrix<f64>, length: usize) -> String {
        let mut gen: Vec<Matrix<f64>> = Vec::new();
        gen.push(seed);

        let mut H0: Matrix<f64> = Matrix::zeros(1, hdim);

        for i in 0..length {
            let mut C = &gen[i] * Wc + &H0 * Uc;
            for v in C.mut_data().iter_mut() {
                *v = v.tanh();
            }

            let mut I = &gen[i] * Wi + &H0 * Ui;
            for v in I.mut_data().iter_mut() {
                let val = *v;
                *v = 1. / (1. + std::f64::consts::E.powf(val * -1.));
            }

            let mut F = &gen[i] * Wf + &H0 * Uf;
            for v in F.mut_data().iter_mut() {
                let val = *v;
                *v = 1. / (1. + std::f64::consts::E.powf(val * -1.));
            }

            let mut H = Matrix::zeros(1, hdim);
            for ((((h, c), i), f), h1) in H.mut_data().iter_mut().zip(C.data().iter()).zip(I.data().iter()).zip(F.data().iter()).zip(H0.data().iter()) {
                *h = (*i * *c + *f * *h1).tanh();
            }

            let mut O = &H * Wo;
            softmax(&mut O);

            gen.push(O);
            H0 = H;
        }

        let mut ret = String::new();
        for g in gen.iter() {
            let c = mtx_to_c(g, etoc);
            ret.push(c);
        }
        ret
    }
    pub fn train(input: Vec<usize>, enc: usize, etoc: HashMap<usize, char>, ctoe: HashMap<char, usize>, hdim: usize, iter: usize, slen: usize, step: f64, decay: f64) -> LSTMTextGen {
        let mut Wc = rmtx(enc, hdim);
        let mut Uc = rmtx(hdim, hdim);
        let mut Wi = rmtx(enc, hdim);
        let mut Ui = rmtx(hdim, hdim);
        let mut Wf = rmtx(enc, hdim);
        let mut Uf = rmtx(hdim, hdim);
        let mut Wo = rmtx(hdim, enc);

        let mut mWc: Matrix<f64> = Matrix::zeros(enc, hdim);
        let mut mUc: Matrix<f64> = Matrix::zeros(hdim, hdim);
        let mut mWi: Matrix<f64> = Matrix::zeros(enc, hdim);
        let mut mUi: Matrix<f64> = Matrix::zeros(hdim, hdim);
        let mut mWf: Matrix<f64> = Matrix::zeros(enc, hdim);
        let mut mUf: Matrix<f64> = Matrix::zeros(hdim, hdim);
        let mut mWo: Matrix<f64> = Matrix::zeros(hdim, enc);

        let data = create_input(&input, enc);

        let mut smooth_loss = -1. * (1. / enc as f64).log(std::f64::consts::E) * slen as f64;

        for k in 0..iter {
            let mut ss = 0;
            let mut Hprev: Matrix<f64> = Matrix::zeros(1, hdim);

            while ss < data.len() - 1 {
                let mut loss = 0.;
                let mut Cs: Vec<Matrix<f64>> = Vec::new();
                let mut Is: Vec<Matrix<f64>> = Vec::new();
                let mut Fs: Vec<Matrix<f64>> = Vec::new();
                let mut Hs: Vec<Matrix<f64>> = Vec::new();
                let mut Os: Vec<Matrix<f64>> = Vec::new();
                Hs.push(Hprev.clone());

                //forward pass
                for i in ss..min(ss + slen, data.len() - 1) {
                    let mut C = &data[i] * &Wc + &Hs[i - ss] * &Uc;
                    for v in C.mut_data().iter_mut() {
                        *v = v.tanh();
                    }

                    let mut I = &data[i] * &Wi + &Hs[i - ss] * &Ui;
                    for v in I.mut_data().iter_mut() {
                        let val = *v;
                        *v = 1. / (1. + std::f64::consts::E.powf(val * -1.));
                    }

                    let mut F = &data[i] * &Wf + &Hs[i - ss] * &Uf;
                    for v in F.mut_data().iter_mut() {
                        let val = *v;
                        *v = 1. / (1. + std::f64::consts::E.powf(val * -1.));
                    }

                    let mut H = Matrix::zeros(1, hdim);

                    for ((((h, c), i), f), h1) in H.mut_data().iter_mut().zip(C.data().iter()).zip(I.data().iter()).zip(F.data().iter()).zip(Hs[i - ss].data().iter()) {
                        *h = (*i * *c + *f * *h1).tanh();
                    }

                    let mut O = &H * &Wo;
                    softmax(&mut O);
                    Os.push(O);
                    Hs.push(H);
                    Fs.push(F);
                    Is.push(I);
                    Cs.push(C);
                }

                //compute loss
                for i in ss..min(ss + slen, data.len() - 1) {
                    let or = Os[i - ss].get_row(0).expect("invalid row index 0 for computed output");
                    let idx = input[i+1];
                    loss += or[idx].log(std::f64::consts::E) * -1.;
                }
                smooth_loss = smooth_loss * 0.999 + loss * 0.001;

                //back propagation
                let mut dWc: Matrix<f64> = Matrix::zeros(enc, hdim);
                let mut dUc: Matrix<f64> = Matrix::zeros(hdim, hdim);
                let mut dWi: Matrix<f64> = Matrix::zeros(enc, hdim);
                let mut dUi: Matrix<f64> = Matrix::zeros(hdim, hdim);
                let mut dWf: Matrix<f64> = Matrix::zeros(enc, hdim);
                let mut dUf: Matrix<f64> = Matrix::zeros(hdim, hdim);
                let mut dWo: Matrix<f64> = Matrix::zeros(hdim, enc);
                let mut Hn : Matrix<f64> = Matrix::zeros(1, hdim);
                let mut j: i64 = min(ss + slen - 1, data.len() - 2) as i64;

                while j >= ss as i64 {
                    let o = &Os[j as usize - ss];
                    let y = &data[j as usize + 1];
                    let H = &Hs[j as usize - ss + 1];
                    let HT1 = &Hs[j as usize - ss];
                    let F = &Fs[j as usize - ss];
                    let I = &Is[j as usize - ss];
                    let C = &Cs[j as usize - ss];

                    let mut dHT1 = Matrix::zeros(1, hdim);
                    
                    let dY = o - y;

                    dWo += H.transpose() * &dY;

                    let mut dH = &dY * Wo.transpose() + &Hn;
                    for (dh, h) in dH.mut_data().iter_mut().zip(H.data().iter()) {
                        let k = (1. - h * h) * *dh;
                        *dh = k;
                    }

                    let mut dC = Matrix::zeros(1, hdim);
                    for ((v, h), i) in dC.mut_data().iter_mut().zip(dH.data().iter()).zip(I.data().iter()) {
                        *v = *h * *i;
                    }
                    for (dc, c) in dC.mut_data().iter_mut().zip(C.data().iter()) {
                        let k = (1. - c * c) * *dc;
                        *dc = k;
                    }
                    dWc += data[j as usize].transpose() * &dC;
                    dUc += HT1.transpose() * &dC;
                    dHT1 += &dC * Uc.transpose();

                    let mut dI = Matrix::zeros(1, hdim);
                    for ((v, h), c) in dI.mut_data().iter_mut().zip(dH.data().iter()).zip(C.data().iter()) {
                        *v = *h * *c;
                    }
                    for (di, i) in dI.mut_data().iter_mut().zip(I.data().iter()) {
                        let k = i * (1. - i) * *di;
                        *di = k;
                    }
                    dWi += data[j as usize].transpose() * &dI;
                    dUi += HT1.transpose() * &dI;
                    dHT1 += &dI * Ui.transpose();

                    let mut dF = Matrix::zeros(1, hdim);
                    for ((v, dh), ht1) in dF.mut_data().iter_mut().zip(dH.data().iter()).zip(HT1.data().iter()) {
                        *v = *dh * *ht1;
                    }
                    for (df, f) in dF.mut_data().iter_mut().zip(F.data().iter()) {
                        let k = f * (1. - f) * *df;
                        *df = k;
                    }
                    dWf += data[j as usize].transpose() * &dF;
                    dUf += HT1.transpose() * &dF;
                    dHT1 += &dF * Uf.transpose();

                    Hn = dHT1;

                    j -= 1;
                }

                //regularization
                clip(&mut dWc);
                clip(&mut dUc);
                clip(&mut dWi);
                clip(&mut dUi);
                clip(&mut dWf);
                clip(&mut dUf);
                clip(&mut dWo);

                //RMSProp parameter update
                adagrad(&mut Wc, &mut dWc, &mut mWc, step);
                adagrad(&mut Uc, &mut dUc, &mut mUc, step);
                adagrad(&mut Wi, &mut dWi, &mut mWi, step);
                adagrad(&mut Ui, &mut dUi, &mut mUi, step);
                adagrad(&mut Wf, &mut dWf, &mut mWf, step);
                adagrad(&mut Uf, &mut dUf, &mut mUf, step);
                adagrad(&mut Wo, &mut dWo, &mut mWo, step);

//                rmsprop(&mut Wc, &mut dWc, &mut mWc, step, decay);
//                rmsprop(&mut Uc, &mut dUc, &mut mUc, step, decay);
//                rmsprop(&mut Wi, &mut dWi, &mut mWi, step, decay);
//                rmsprop(&mut Ui, &mut dUi, &mut mUi, step, decay);
//                rmsprop(&mut Wf, &mut dWf, &mut mWf, step, decay);
//                rmsprop(&mut Uf, &mut dUf, &mut mUf, step, decay);
//                rmsprop(&mut Wo, &mut dWo, &mut mWo, step, decay);

                Hprev = Hs[Hs.len() - 1].clone();
                ss += slen;
            }

            if k % 100 == 0 {
                println!("{} = {}", k, smooth_loss);
                let seed = data[0].clone();
                println!("{}", Self::sample(&Wc, &Uc, &Wi, &Ui, &Wf, &Uf, &Wo, &etoc, hdim, seed, input.len()));
            }
        }

        LSTMTextGen{ Wc: Wc, Uc: Uc, Wi: Wi, Ui: Ui, Wf: Wf, Uf: Uf, Wo: Wo, ctoe: ctoe, etoc: etoc }
    }
}
