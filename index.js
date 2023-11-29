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

                console.log('Image Data Length:', imageData.length);

                for (let i = 0; i < imageData.length; i += 4) {
                    const r = imageData[i] / 255;
                    const g = imageData[i + 1] / 255;
                    const b = imageData[i + 2] / 255;

                    tensorData[i / 4] = r;
                    tensorData[(i / 4) + (48 * 48)] = g;
                    tensorData[(i / 4) + (2 * 48 * 48)] = b;
                }

                const expectedDims = [1, 3, 48, 48];
                console.log('Expected Dims:', expectedDims);

                const tensor = new onnx.Tensor('float32', tensorData, expectedDims);

                console.log('Reshaped Tensor Data Length:', tensor.data.length);

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
    // Assuming outputTensor is a Map with numeric keys and Float32Array values
    const predictedEmotion = getPredictedEmotion(outputTensor);

    return predictedEmotion;
}

// Function to get the predicted emotion from the output tensor
function getPredictedEmotion(outputTensor) {
    let maxIndex = -1;
    let maxScore = Number.NEGATIVE_INFINITY;

    // Iterate through the Map entries to find the index with the maximum score
    for (const [key, value] of outputTensor.entries()) {
        const scores = Array.from(value);
        const maxScoreIndex = scores.indexOf(Math.max(...scores));

        if (scores[maxScoreIndex] > maxScore) {
            maxScore = scores[maxScoreIndex];
            maxIndex = maxScoreIndex;
        }
    }

    // Replace emotionClasses with your actual class names
    const emotionClasses = ["Happy", "Sad"];

    // Return the predicted emotion
    return emotionClasses[maxIndex];
}

// Declare the ONNX session outside the runInference function
let session;

// Function to run inference on the uploaded image
async function runInference() {
    // Get the input image from the file input
    const fileInput = document.getElementById("imageInput");
    const image = fileInput.files[0];

    try {
        // Convert the image to a tensor (without normalization)
        const tensor = await convertImageToTensor(image);

        // Check if the ONNX session is initialized
        if (!session) {
            // If not, create and load the ONNX session
            session = await onnx.InferenceSession.create('emotion_recognition_model.onnx');
        }

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
