FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3-pip \
        ffmpeg \
        libsm6 \
        libxext6

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

RUN pip install --user -e .

# Again, test and train scripts should be executable within the container.
RUN chmod +x scripts/run_train_dino_score.sh
RUN chmod +x scripts/run_train_dino_sub.sh
RUN chmod +x scripts/run_inference.sh