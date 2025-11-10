FROM porepy/dev

# Set home directory
ENV HOME=/workdir

# Set environment variables
ENV PYGEON_HOME=${HOME}/pygeon \
    PYTHONPATH=${HOME}/pygeon:$PYTHONPATH

# Set working directory
WORKDIR ${HOME}

# Clone the PyGeoN repository
RUN git clone https://github.com/compgeo-mox/pygeon.git ${PYGEON_HOME}

# Install PyGeoN in editable mode
WORKDIR ${PYGEON_HOME}
RUN pip install --no-cache-dir -e .

# Set default working directory
WORKDIR ${PYGEON_HOME}

# Default command opens a bash shell
CMD ["/bin/bash"]