# Use the official Ubuntu image as a base
FROM --platform=linux/amd64 ubuntu:latest

# Set the working directory
ARG USERNAME=user
ARG HOME_DIR=/home/${USERNAME}
ENV HOME=${HOME_DIR}
WORKDIR ${HOME}

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install sudo
RUN sed -i 's/^# deb/deb/' /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get install -y \
    wget git bash libc6 apt-utils build-essential gfortran bzip2 ca-certificates \
    libglu1-mesa ffmpeg libsm6 libxext6 libxrender1 libxcursor1 libxft2 libfontconfig1
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Use bash shell instead
SHELL ["/bin/bash", "-c"]

# Create a non-root user and add to sudoers
RUN useradd -m -d ${HOME_DIR} ${USERNAME}
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${USERNAME}:${USERNAME} ${HOME_DIR}

# Switch to the non-root user
USER ${USERNAME}

# Download and install Miniconda
ENV CONDA_DIR=${HOME}/conda
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
    
# Add Conda binaries to PATH environment variable
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize Conda
RUN conda init bash

# Accept Anaconda license terms for main and r channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create a new environment
ENV PYTHON_VERSION=3.13

RUN conda create -y -n py3 python=${PYTHON_VERSION}
SHELL ["conda", "run", "-n", "py3", "/bin/bash", "-c"]
RUN echo ". activate py3" >> ${HOME}/.bashrc
RUN . ${HOME}/.bashrc && conda install -y jupyter

# Expose the Jupyter Notebook port
EXPOSE 8888

# Clone the repository
RUN cd ${HOME} && git clone https://github.com/compgeo-mox/pygeon.git
RUN cd ${HOME}/pygeon && pip install --no-cache-dir -e .

# Set the default command to start the environment
CMD [ "/bin/bash", "-c", ". activate py3" ]