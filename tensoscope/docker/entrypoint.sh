#!/bin/sh
# Normalise BASE_PATH: ensure leading slash, strip trailing slash
: "${BASE_PATH:=/}"
case "$BASE_PATH" in
  /*) ;;
  *) BASE_PATH="/$BASE_PATH" ;;
esac
BASE_PATH="${BASE_PATH%/}"

# Create writable nginx runtime dirs (needed for rootless containers)
mkdir -p \
  /tmp/nginx/conf.d \
  /tmp/nginx/client_temp \
  /tmp/nginx/proxy_temp \
  /tmp/nginx/fastcgi_temp \
  /tmp/nginx/uwsgi_temp \
  /tmp/nginx/scgi_temp

if [ -z "$BASE_PATH" ]; then
    cp /etc/tensogram/nginx.root.template /tmp/nginx/conf.d/default.conf
    echo "Serving at BASE_PATH=/"
else
    export BASE_PATH
    envsubst '$BASE_PATH' \
      < /etc/tensogram/nginx.conf.template \
      > /tmp/nginx/conf.d/default.conf
    echo "Serving at BASE_PATH=$BASE_PATH"
fi
