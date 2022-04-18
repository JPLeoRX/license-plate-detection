FROM python:3.8

# Update & system tools
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 -y

# Dependencies
RUN pip install fastapi==0.63.0
RUN pip install numpy==1.19.4
RUN pip install opencv-python==4.5.1.48
RUN pip install simplestr==0.5.0
RUN pip install --no-cache-dir torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Detectron
RUN mkdir /detectron2_temp
WORKDIR /detectron2_temp
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python -m pip install -e detectron2

# Main test script
COPY . /app
WORKDIR /app
RUN mkdir test_output
RUN chmod a+x docker-main.sh
CMD './docker-main.sh'