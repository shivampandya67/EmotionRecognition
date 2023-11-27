function convertImageToTensor(image) {
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
                const canvas = document.createElement('canvas');
                canvas.width = 48;
                canvas.height = 48;
                const ctx = canvas.getContext('2d');

                ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                const normalizedData = imageData.map(value => value / 255.0);

                // Assuming your model expects the input size [1, 3, 48, 48]
                const expectedDims = [1, 3, 48, 48];

                // Reshape the normalized data to match the expected input size
                const reshapedData = new Float32Array(expectedDims.reduce((a, b) => a * b, 1));
                normalizedData.forEach((value, index) => {
                    reshapedData[index % reshapedData.length] += value;
                });

                resolve(reshapedData);
            };
        };

        reader.onerror = function (error) {
            reject(error);
        };

        reader.readAsDataURL(image);
    });
}

// Load the ONNX model
const session = new onnx.InferenceSession();
session.loadModel("emotion_recognition_model.onnx");

// Function to run inference on the uploaded image
async function runInference() {
    try {
        // Ensure the session is initialized
        await session.initialize();

        // Get the input image from the file input
        const fileInput = document.getElementById("imageInput");
        const image = fileInput.files[0];

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
