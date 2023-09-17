use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        Dropout,
        Linear,
        ReLU, DropoutConfig, LinearConfig, loss::CrossEntropyLoss,
    },
    tensor::{backend::Backend, Tensor, activation::softmax, Float, Int},
    train::{ClassificationOutput}
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub conv1: Conv2d<B>,
    pub active1: ReLU,
    pub pool1: MaxPool2d,
    pub dropout1: Dropout,
    pub active2: ReLU,
    pub conv2: Conv2d<B>,
    pub pool2: MaxPool2d,
    pub dropout2: Dropout,
    pub linear1: Linear<B>,
    pub active3: ReLU,
    pub dropout3: Dropout,
    pub linear2: Linear<B>,
}

const NUM_CLASSES: usize = 10;

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub num_classes: usize,
    pub hidden_size: usize,
    #[config(default = "0.5")]
    pub dropout1: f64,
    pub dropout2: f64,
    pub dropout3: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        let model = Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            active1: ReLU::new(),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            dropout1: DropoutConfig::new(self.dropout1).init(),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            active2: ReLU::new(),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            dropout2: DropoutConfig::new(self.dropout2).init(),
            linear1: LinearConfig::new(16 * 22 * 22, self.hidden_size).init(),
            active3: ReLU::new(),
            dropout3: DropoutConfig::new(self.dropout3).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
        };

        model
    }
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, channels, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.active1.forward(x);
        let x = self.pool1.forward(x);
        let x = self.dropout1.forward(x);
        let x = self.conv2.forward(x); 
        let x = self.active2.forward(x);
        let x = self.pool2.forward(x);
        let x = self.dropout1.forward(x);
        let x = x.flatten(1, 3);
        let x = self.linear1.forward(x);
        let x = self.active3.forward(x);
        let x = self.dropout3.forward(x);

        x
    }

    pub fn forward_classification(&self, images: Tensor<B, 4>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let out = self.forward(images);
        let output = softmax(out, 1);
        let loss = CrossEntropyLoss::default().forward(output.clone(), targets.clone());

        ClassificationOutput { 
            loss,
            output,
            targets,
        }
    }
}
