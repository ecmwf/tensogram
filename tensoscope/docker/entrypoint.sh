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

# Use the container's actual DNS nameserver so nginx can resolve upstream
# hosts. 8.8.8.8 is often unreachable inside restricted container networks.
RESOLVER=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf)
: "${RESOLVER:=127.0.0.11}"
export RESOLVER

if [ -z "$BASE_PATH" ]; then
    envsubst '$RESOLVER' \
      < /etc/tensogram/nginx.root.template \
      > /tmp/nginx/conf.d/default.conf
    echo "Serving at BASE_PATH=/"
else
    export BASE_PATH
    envsubst '$BASE_PATH $RESOLVER' \
      < /etc/tensogram/nginx.conf.template \
      > /tmp/nginx/conf.d/default.conf
    echo "Serving at BASE_PATH=$BASE_PATH"
fi
