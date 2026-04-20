# Configurable via env or command line:
#   make build
#   BASE_PATH=/subpath make run
#   TAG=1.2.3 REGISTRY=my.registry/tensoscope make push
#
# ECMWF users — push to ECCR:
#   TAG=1.2.3 REGISTRY=eccr.ecmwf.int/tensoscope make push
IMAGE_NAME ?= tensoscope
TAG        ?= latest
BASE_PATH  ?= /
REGISTRY   ?=

LOCAL_IMAGE := $(IMAGE_NAME):$(TAG)

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

## Tag and push to REGISTRY (requires prior: make build, and REGISTRY to be set)
push:
	@test -n "$(REGISTRY)" || (echo "REGISTRY is not set. Example: REGISTRY=eccr.ecmwf.int/tensoscope make push" && exit 1)
	podman tag $(LOCAL_IMAGE) $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	podman push $(REGISTRY)/$(IMAGE_NAME):$(TAG)

## Run the container locally (exposes port 8000)
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
