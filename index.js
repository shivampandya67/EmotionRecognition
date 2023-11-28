// Function to preview the uploaded image
function previewImage(event) {
  const preview = document.getElementById('preview');
  const file = event.target.files[0];
  const reader = new FileReader();

  reader.onloadend = function() {
    preview.src = reader.result;
    preview.style.display = 'block';
  }

  if (file) {
    reader.readAsDataURL(file);
  } else {
    preview.src = '';
    preview.style.display = 'none';
  }
}

// Function to run inference on the uploaded image
async function runInference() {
  const preview = document.getElementById('preview');
  if (!preview.src || preview.src === '#') {
    alert('Please upload an image first.');
    return;
  }

  const tensorX = await preprocessImage(preview); // Preprocess the image

  try {
    const session = await ort.InferenceSession.create('EmotionCnnOnnx.onnx');
    const result = await session.run({ input: tensorX });
    const outputData = result.output.data;

    // Process outputData to get predicted class, then display the results
    displayResults(outputData); // Function to display results (defined below)
  } catch (error) {
    console.error('Error running inference:', error);
    alert('Failed to run inference. Please try again.');
  }
}

// Function to preprocess the uploaded image
async function preprocessImage(imageElement) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  canvas.width = 48;
  canvas.height = 48;

  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

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

  const tensorX = new ort.Tensor('float32', tensorData, [1, 3, 48, 48]);

  return tensorX;
}

// Function to display the inference results
function displayResults(outputData) {
  // Example display of results
??
  const classifiedImage = document.getElementById('classifiedImage');
  const classificationInfo = document.getElementById('classificationInfo');

  classifiedImage.src = document.getElementById('preview').src;
  classificationInfo.innerText = `Actual Class: ${actualClass}, Predicted Class: ${predictedClass}, Correct: ${isCorrect}`;
}
