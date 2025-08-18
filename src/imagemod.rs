use std::sync::Arc;
use ort::{environment::Environment, session::Session, session::SessionBuilder, Value, LoggingLevel};
use ndarray::{Array4, CowArray};
use anyhow::Result;
use image::ImageReader;



pub struct OrtImageClassifier {
    pub session: Session,
    #[allow(dead_code)]
    pub environment: Arc<Environment>,
}
impl OrtImageClassifier {
    pub fn new(model_path: &str) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("classify")
                .with_log_level(LoggingLevel::Warning)
                .build()?
        );

        let session = SessionBuilder::new(&Arc::new(environment.as_ref().clone()))?
            .with_model_from_file(model_path)?;

        Ok(Self { session, environment })
    }
    // pub fn classify(session: &Session, input_tensor: Array4<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // pub fn classify(&self, input_tensor: Array4<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    pub fn classify(classifier: &Self, input_tensor: Array4<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Convert the ndarray array to a CowArray
        let cow_array = CowArray::from(input_tensor).into_dyn();

        let allocator_ptr = classifier.session.allocator();

        // let cow_array = CowArray::from(&input_tensor).into_dyn();
        let input_value = Value::from_array(allocator_ptr.clone(), &cow_array)?;
        // let input_value = Value::from_array(allocator_ptr, &cow_array)?;

        // Run inference
        // let outputs = &self.session.run(vec![input_value])?;
        let outputs = classifier.session.run(vec![input_value])?;

        // Extract the output as a Vec<f32>
        let output_array = outputs[0].try_extract::<f32>()?;
        Ok(output_array.view().iter().cloned().collect())
    }
}

pub fn preprocess_image(path: &str) -> Array4<f32> {
    // Load image from file
    let img = ImageReader::open(path)
        .expect("Failed to open image")
        .decode()
        .expect("Failed to decode image")
        .resize_exact(224, 224, image::imageops::FilterType::Triangle);

    let rgb = img.to_rgb8(); // Ensure 3 channels

    // ImageNet normalization
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    // Prepare output tensor: [batch, channels, height, width]
    let mut array = Array4::<f32>::zeros((1, 3, 224, 224));

    for (y, x, pixel) in rgb.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
        array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
        array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
    }

    array
}

pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|x| x / sum).collect()
}

// OrtOwnedTensor<f32, IxDyn>  (ONNX output tensor)
//            |
//            | .view()   -> creates a lightweight read-only view of the tensor
//            v
// ArrayViewD<f32>  (ndarray view)
//            |
//            | .iter()   -> iterator over &f32 references
//            v
// Iterator<&f32>
//            |
//            | .cloned() -> converts &f32 -> f32
//            v
// Iterator<f32>
//            |
//            | .collect() -> collects into a Vec<f32>
//            v
// Vec<f32>  (Rust-owned vector of floats)
// https://huggingface.co/lmz/candle-onnx/blob/main/squeezenet1.1-7.onnx

// https://github.com/onnx/models/blob/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz

// https://github.com/onnx/models/blob/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx