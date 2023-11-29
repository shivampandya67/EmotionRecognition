// Function to convert an image to a tensor
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
                const canvas = document.createElement('canvas');
                canvas.width = 48;
                canvas.height = 48;
                const ctx = canvas.getContext('2d');

                ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
                const tensorData = new Float32Array(3 * 48 * 48);

                for (let i = 0; i < imageData.length; i += 4) {
                    const r = imageData[i] / 255;
                    const g = imageData[i + 1] / 255;
                    const b = imageData[i + 2] / 255;

                    tensorData[i / 4] = r;
                    tensorData[(i / 4) + (48 * 48)] = g;
                    tensorData[(i / 4) + (2 * 48 * 48)] = b;
                }

                const tensor = new onnx.Tensor('float32', tensorData, [1, 3, 48, 48]);

                resolve(tensor);
            };
        };

        reader.onerror = function (error) {
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

// Function to run inference on the uploaded image
async function runInference() {
    // Get the input image from the file input
    const fileInput = document.getElementById("imageInput");
    const image = fileInput.files[0];

    try {
        // Convert the image to a tensor (without normalization)
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

// Add event listener for the file input change
document.getElementById("imageInput").addEventListener("change", runInference);
