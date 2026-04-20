# Configurable via env or command line:
#   make build
#   BASE_PATH=/subpath make run
#   REPOSITORY=my-app TAG=1.2.3 make push
ECCR_REGISTRY ?= eccr.ecmwf.int/tensoscope
IMAGE_NAME    ?= tensoscope
TAG           ?= latest
BASE_PATH     ?= /

LOCAL_IMAGE := $(IMAGE_NAME):$(TAG)
ECCR_IMAGE  := $(ECCR_REGISTRY)/$(IMAGE_NAME):$(TAG)

# Build context is the repo root so both typescript/ and tensoscope/ are accessible
BUILD_CONTEXT := ..

.PHONY: build push run stop logs

## Build the image locally
build:
	podman build \
	  -f Dockerfile \
	  -t $(LOCAL_IMAGE) \
          --platform linux/amd64 \
	  $(BUILD_CONTEXT)

## Tag and push to ECCR (requires prior: make build)
push:
	podman tag $(LOCAL_IMAGE) $(ECCR_IMAGE)
	podman push $(ECCR_IMAGE)

## Run the container locally (exposes port 8080)
run:
	podman run --rm \
	  -p 8000:8000 \
	  -e BASE_PATH=$(BASE_PATH) \
	  --name $(IMAGE_NAME) \
	  $(LOCAL_IMAGE)

## Stop the running container
stop:
	podman stop $(IMAGE_NAME) 2>/dev/null || true

## Follow container logs
logs:
	podman logs -f $(IMAGE_NAME)
