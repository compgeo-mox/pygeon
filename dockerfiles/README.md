# PyGeoN Docker

[![Image Size](https://img.shields.io/docker/image-size/pygeon/main/latest)](https://hub.dockerin)

## Usage
Pull the image:
```bash
docker pull pygeon/main
```

Run a container:
```bash
docker run -it --rm pygeon
```

## Using Docker Directly

Build the image:
```bash
docker build -t pygeon -f dockerfiles/Dockerfile .
```

## Using VS Code Dev Containers

1. Install the "Dev Containers" extension in VS Code
2. Open the PyGeoN project in VS Code
3. Press `F1` and select "Dev Containers: Reopen in Container"
4. VS Code will build the container and set up the development environment automatically
