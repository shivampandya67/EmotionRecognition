// Load the ONNX model
const session = new onnx.InferenceSession();
session.loadModel("emotion_recognition_model.onnx");

function convertImageToTensor(image) {
    return new Promise((resolve, reject) => {
        // Create an HTML image element
        const imgElement = new Image();
        imgElement.src = URL.createObjectURL(image);

        // Wait for the image to load
        imgElement.onload = () => {
            // Create a canvas element to draw the image
            const canvas = document.createElement('canvas');
            canvas.width = 48; // Adjust the width according to your model input size
            canvas.height = 48; // Adjust the height according to your model input size
            const ctx = canvas.getContext('2d');

            // Draw the image on the canvas
            ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

            // Get the pixel data from the canvas
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

            // Normalize the pixel values to be in the range [0, 1]
            const normalizedData = imageData.map(value => value / 255.0);

            // Create a tensor from the normalized data
            const tensor = new onnx.Tensor(new Float32Array(normalizedData), 'float32', [1, 3, canvas.width, canvas.height]);

            // Resolve with the tensor
            resolve(tensor);
        };

        // Handle image load errors
        imgElement.onerror = (error) => {
            reject(error);
        };
    });
}



// Function to run inference on the uploaded image
async function runInference() {
    // Get the input image from the file input
    const fileInput = document.getElementById("imageInput");
    const image = fileInput.files[0];

    // Preprocess the image (convert to tensor, resize, normalize, etc.)
    const tensor = await preprocessImage(image);

    // Run inference
    const outputTensor = await session.run([tensor]);

    // Process the output (interpret the result)
    const predictedEmotion = processOutput(outputTensor);

    // Display the result on the webpage
    const outputDiv = document.getElementById("output");
    outputDiv.innerHTML = `Predicted Emotion: ${predictedEmotion}`;
}

// Function to preprocess the image
async function preprocessImage(image) {
    // Convert the image to a tensor (implement your logic here)
    // Example: Assuming you have a function to convert image to tensor
    const tensor = await convertImageToTensor(image);

    // Implement any required preprocessing steps

    return tensor;
}

// Function to process the output tensor
function processOutput(outputTensor) {
    // Implement logic to interpret the output tensor
    // Example: Assuming outputTensor is a Float32Array with confidence scores
    const maxIndex = outputTensor.indexOf(Math.max(...outputTensor));
    const emotionClasses = ["Happy", "Sad"];  // Replace with your actual class names
    const predictedEmotion = emotionClasses[maxIndex];

    return predictedEmotion;
}
