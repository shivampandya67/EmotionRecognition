// Load the ONNX model
const session = new onnx.InferenceSession();
session.loadModel("emotion_recognition_model.onnx");

// Function to run inference on the uploaded image
async function runInference() {
    // Get the input image from the file input
    const fileInput = document.getElementById("imageInput");
    const image = fileInput.files[0];

    try {
        // Preprocess the image (convert to tensor, resize, normalize, etc.)
        const tensor = await convertImageToTensor(image);

        // Run inference
        const outputTensor = await session.run([tensor]);

        // Process the output (interpret the result)
        const predictedEmotion = processOutput(outputTensor);

        // Display the result on the webpage
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = `Predicted Emotion: ${predictedEmotion}`;
    } catch (error) {
        console.error("Error during inference:", error);
    }
}

// Function to preprocess the image
async function convertImageToTensor(image) {
    return new Promise((resolve, reject) => {
        if (!(image instanceof Blob)) {
            reject(new Error('Invalid image parameter'));
            return;
        }

        const reader = new FileReader();

        reader.onload = function (e) {
            const imgElement = new Image();
            imgElement.src = e.target.result;

            imgElement.onload = () => {
                console.log('Image loaded successfully.');

                const canvas = document.createElement('canvas');
                canvas.width = 48;
                canvas.height = 48;
                const ctx = canvas.getContext('2d');

                ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

                // Assuming your model expects the input size [1, 3, 48, 48]
                const expectedDims = [1, 3, 48, 48];

                console.log('Image Data Length:', imageData.length);
                console.log('Expected Dims:', expectedDims);

                // Ensure that the tensor data length matches the expected input size
                if (imageData.length !== expectedDims.reduce((a, b) => a * b, 1)) {
                    console.error('Error: Input dims do not match data length');
                    reject(new Error('Input dims do not match data length'));
                    return;
                }

                // Convert the data to a Float32Array
                const tensorData = new Float32Array(imageData);

                console.log('Reshaped Tensor Data Length:', tensorData.length);

                const tensor = new onnx.Tensor(tensorData, 'float32', expectedDims);

                console.log('Tensor created successfully.');
                resolve(tensor);
            };
        };

        reader.onerror = function (error) {
            console.error('Error reading image:', error);
            reject(error);
        };

        reader.readAsDataURL(image);
    });
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
