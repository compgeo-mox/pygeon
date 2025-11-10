FROM porepy/dev

# Set environment variables
ENV PYGEON_HOME=${HOME}/pygeon

# Set working directory
WORKDIR ${HOME}

# Clone the PyGeoN repository
RUN git clone https://github.com/compgeo-mox/pygeon.git ${PYGEON_HOME}
WORKDIR ${PYGEON_HOME}

# Install additional dependencies if needed
RUN pip install -e .

# Add PyGeoN home to the pythonpath. This may or may not be necessary.
ENV PYTHONPATH $PYGEON_HOME:$PYTHONPATH

# Run tests to check that everything works.
WORKDIR ${PYGEON_HOME}/tests
RUN pytest

# Move back to code directory
WORKDIR ${HOME}