FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory inside the container
WORKDIR /workspace

# Set environment variables
ENV PATH /usr/local/cuda/bin:$PATH
ENV FORCE_CUDA 1
ENV PYTHONPATH ${PYTHONPATH}:/workspace

# Copy the required files of the project into the container at /workspace/
COPY requirements.txt /workspace/
COPY pytorch-quantization /workspace/pytorch-quantization/
COPY docker/banner.sh /etc/banner.sh

# Install any additional dependencies specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    cd pytorch-quantization/ && \
    python setup.py install && \
    apt-get install -y curl && cd ../ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/

EXPOSE 8888

# Label the image
LABEL version="1.0" \
      maintainer="Dimitrios Danopoulos dimdano@microlab.ntua.gr"


