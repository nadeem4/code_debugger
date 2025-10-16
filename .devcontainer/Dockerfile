# syntax=docker/dockerfile:1.4

ARG VARIANT="3.11"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure the workspace uses UTF-8 locales
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

