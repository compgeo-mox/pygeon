# PyGeoN Docker Setup

This directory contains Docker configurations for PyGeoN development.

## Using Docker Directly

Build the Docker image:
```bash
docker build -t pygeon:latest -f dockerfiles/Dockerfile .
```

Run a container:
```bash
docker run -it --rm pygeon:latest
```

## Using VS Code Dev Containers

1. Install the "Dev Containers" extension in VS Code
2. Open the PyGeoN project in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. VS Code will build the container and set up the development environment automatically