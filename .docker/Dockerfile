
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY .. .

RUN mkdir -p build && cd build && cmake ..

# Build the project
RUN cd build && make

CMD ["./build/SmartDNN"]