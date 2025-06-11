# WanVideo_Serverless_Runpod
A serverless application that uses WAN 2.1 to run a Text-to-Video task on RunPod.

## 1. Introduction
This is a serverless application that uses [Wan2.1](https://github.com/Wan-Video/Wan2.1) to run a **Text-to-Video** task on [RunPod](https://www.runpod.io/).

Specifically, this repo is mainly modified from [kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper), which costs much less VRAM and is much faster (thanks to CausVid Lora) than the official repo.

Serverless means that you are only charged for the time you use the application, and you don't need to pay for idle time, which is very suitable for applications that are not used frequently but need to respond quickly.

Theoretically, this application can be called by any other application. Here we provide one example:
1. A simple Python script

See [Usage](#Usage) below for more details.

### Example Result:
Input: see test_input.json for more details

Result:
(6 steps, 832x480 resolution, 97 frames, takes ~140 seconds on an RTX 4090, and costs about $0.045 on RunPod.)

https://github.com/user-attachments/assets/cd847d26-70e4-4fad-94c0-cfb62f1b0c1f

#### Time Measurement Explanation:
The time is measured from the moment the input prompt is sent to the moment the result image is received, including the time for all the following steps:
- Receive the request from the client
- Serverless container startup
- Model loading
- Inference
- Sending the result image back to the client.

## 2. Dependencies
- Python >= 3.10
- Docker
- A local GPU is necessary for testing, but not necessary for deployment. (Recommended: RTX 4090, A100)

Example Notebook: [link](./scripts/WanVideo_Serverless_Runpod.ipynb)

<a id="Usage"></a>
## 3. Usage
#### 1. Test on Local Machine
```
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download.py

# Run inference test
python wan_inference_utils.py

# Run server.py local test
python server.py
```

#### 2. Deploy on RunPod
1. First, make sure you have installed Docker and have accounts on both DockerHub and RunPod.

2. Then, decide a name for your Docker image, e.g., "your_username/wan2_1:v1", and set your image name in "./scripts/build.sh".

3. Run the following commands to build and push your Docker image to DockerHub.

bash scripts/build.sh


4. Finally, deploy your application on RunPod to create [Template](https://docs.runpod.io/docs/template-creation) and [Endpoint](https://docs.runpod.io/docs/autoscaling).

You can find many detailed instructions on Google about how to deploy a Docker image on RunPod.

Feel free to contact the author of this repo if you encounter any problems after searching on Google.

#### 3. Call the Application
##### Call from a Python script
```
# Make sure to set API key and endpoint ID before running the script.
python test_client.py
```

## 4. TODO
- [x] Support for simple Text-to-Video task
- [x] Create a notebook example
- [ ] Prevent errors when using RTX 5090
- [ ] Support for reference image integration using VACE

## 4. Acknowledgement
Thanks to [kijai's ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper), [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [RunPod](https://www.runpod.io/).
