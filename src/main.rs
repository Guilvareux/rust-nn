use burn::{
    optim::{AdamConfig, Optimizer, GradientsParams, decay::WeightDecayConfig},
    tensor::{Tensor, Float, Shape, Data, Int},
    autodiff::ADBackendDecorator,
    nn::loss::CrossEntropyLoss,
    module::Module,
};

use burn::nn as nn;
//use burn_tch::{TchBackend, TchDevice};
use burn_wgpu::{WgpuDevice, WgpuBackend, Vulkan};
use model::{Model, ModelConfig};
use polars::prelude::*;
use plotly::Scatter;
use plotly as plt;
mod model;

const EPOCH: usize = 20;
const BATCH_SIZE: usize = 128;
const LEARN_RATE: f64 = 0.00001;

fn main() {
    /*
    let instance = wgpu::Instance::default();
    {
        println!("Available adapters:");
        for a in instance.enumerate_adapters(wgpu::Backends::all()) {
            println!("    {:?}", a.get_info())
        }
    }
    */

    type GBackend = ADBackendDecorator<WgpuBackend<Vulkan, f32, i32>>;
    //type GBackend = ADBackendDecorator<TchBackend<f32>>;

    //let device = TchDevice::Cuda(0);
    let device = WgpuDevice::DiscreteGpu(0);
    //let device = WgpuDevice::default();
    let train = CsvReader::from_path("../../datasets/cnn/train.csv").unwrap().has_header(true).finish().unwrap();
    //let labels = train.column("label").unwrap().i64().unwrap();
    let labels = train.column("label").unwrap().i64().unwrap().cast(&DataType::Int32).unwrap();
    let train = train.drop("label").unwrap();
    let nlabels = labels.i32().unwrap().to_ndarray().unwrap().to_vec();
    //let nlabels = labels.to_ndarray::<Int32Type>().unwrap().into_raw_vec();
    //let nlabels = labels.to_ndarray().unwrap().to_vec();
    let ndata = train.to_ndarray::<Float32Type>(IndexOrder::C).unwrap();
    let v = ndata.into_raw_vec();
    //let d_len = (v.len() / 784) / BATCH_SIZE as usize;
    let d = Data::new(v.clone(), Shape::new([v.len(), 1, 28, 28]));
    let l = Data::new(nlabels.clone(), Shape::new([v.len()]));
    let l_data: Tensor<GBackend, 1, Int> = Tensor::from_data_device(l, &device);
    let t_data: Tensor<GBackend, 4, Float> = Tensor::from_data_device(d, &device);

    let loss_fn: CrossEntropyLoss<GBackend> = nn::loss::CrossEntropyLoss::new(None).to_device(&device);

    let model: Model<GBackend> = ModelConfig {num_classes: 10, hidden_size: 128, dropout1: 0.5, dropout2: 0.5, dropout3: 0.5}.init();
    let mut model = model.to_device(&device);

    let mut adam = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))).init::<GBackend, Model<GBackend>>();

    let trace1 = Scatter::new(vec![1, 2, 3, 4], vec![10, 15, 13, 17])
        .name("trace1")
        .mode(plotly::common::Mode::Markers);
    let trace2 = Scatter::new(vec![2, 3, 4, 5], vec![16, 5, 11, 9])
        .name("trace2")
        .mode(plotly::common::Mode::Lines);
    let trace3 = Scatter::new(vec![1, 2, 3, 4], vec![12, 9, 15, 12]).name("trace3");

    /*
    let mut plt = plotly::Plot::new();
    plt.add_trace(trace1);
    plt.add_trace(trace2);
    plt.add_trace(trace3);
    plt.show();
    */
    println!("Ready!");
    let stdin = std::io::stdin();
    stdin.read_line(&mut String::new());

    for epoch in 0..EPOCH {
        for i in 0..(v.len()/BATCH_SIZE) {
            let train_data: Tensor<GBackend, 4, Float> = t_data.clone().slice([i*BATCH_SIZE..(((i+1)*BATCH_SIZE)-1)]);
            let target_data: Tensor<GBackend, 1, Int> = l_data.clone().slice([i*BATCH_SIZE..(((i+1)*BATCH_SIZE)-1)]);
            let out = model.forward(train_data);
            let loss = loss_fn.forward(out, target_data);
            println!("Loss: {}", loss);
            //println!("Out: {}", out);
            //println!("Target: {}", target_data);
            //stdin.read_line(&mut String::new());

            let loss = loss.backward();
            let grads = GradientsParams::from_grads(loss, &model);
            model = adam.step(LEARN_RATE, model.clone(), grads);
        }
    }
}
