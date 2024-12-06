# 🎥 REVEX: Removal-Based Video Explainability 🚀
This repository contains the implementation of REVEX, based on the paper "REVEX: A Unified Framework for Removal-Based Explainable Artificial Intelligence in Video".

If you've ever used LIME to explain predictions for images or text, think of REVEX as LIME for videos! 🎉 It helps you understand how your model makes decisions on video data by perturbing parts of the input and analyzing the effects.

---
# 🚀 Quick Start
Here’s how you can start using Video Perturbation Analyzer in just a few steps:

1️⃣ Clone the Repository
```
git clone https://github.com/DaviMPaiva/Lime4video.git
```
2️⃣ Instantiate the Main Class
Create an instance of the VideoPerturbationAnalyzer:

``` python
analyzer = VideoPerturbationAnalyzer()
```
3️⃣ Explain an Instance
Use the explain_instance method to analyze your video. Provide a callback function (your model's prediction function) and some parameters:

```python
analyzer.explain_instance(
    model_function=predict_fn, 
    video_path="input_video.mp4", 
    num_matrix=20, 
    output_folder="output.mp4"
)
````
* 🧩 model_function: A callback function that takes a 3D numpy array (video data) and returns the prediction, a number (e.g., a score).
* 🎞️ video_path: Path to your input video file.
* 🔢 num_matrix: Number of superpixels (fewer superpixels = faster but less precise; I recommend 20).
* 📂 output_folder: Path to save the output annotated video.
### 📋 Example Callback Function
Here’s a simple example of what your predict_fn might look like:

```python

video_path = r""
desired_action = 0

# Load the pretrained model
model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
model.eval()

def predict_fn(frames):
    
    # Transform the video frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transformed_frames = []
    for frame in frames:
        transformed_frame = transform(frame)  # Transform each frame individually
        transformed_frames.append(transformed_frame)
    frames = torch.stack(transformed_frames)
    
    # Reshape to (B, C, T, H, W) format
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)

    frames = frames.to("cuda")
    model.to("cuda")
    with torch.no_grad():
        outputs = model(frames)
    scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return scores[0][desired_action]


if __name__ == '__main__':
    # Example usage:
    analyzer = VideoPerturbationAnalyzer()
    analyzer.explain_instance(model_function=predict_fn, video_path=video_path,
                              num_matrix=num_matrix, output_folder="output.mp4")

```
### 🌟 Why Use This Library?
Explainability Matters: 
    Understand what and where your model focuses on in video inputs.
    Easy Integration: Seamlessly works with your existing video-based ML pipelines.
    Customizable Precision: Tweak num_matrix for the perfect balance between speed and detail.

### 🛠 Suggestions or Questions?
I'd ❤️ to hear from you! Feel free to open an issue or submit a pull request. 

### 📄 License
This project is licensed under the MIT License.

Let’s make video-based AI more interpretable and trustworthy together! 🧑‍💻🎉
