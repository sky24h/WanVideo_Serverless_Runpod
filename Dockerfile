# Include base image
FROM docker.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Define working directory
WORKDIR /workspace/

# Set timezone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 vim && apt-get autoremove -y && apt-get clean -y

# Add models and code
ADD ComfyUI ./ComfyUI

# pip install
ADD requirements.txt ./
RUN pip install -r requirements.txt

# Add necessary files
ADD server.py ./
ADD wan_inference_utils.py ./

# Run server
CMD [ "python", "-u", "./server.py" ]
# # for local testing
# ADD ./test_input.json ./