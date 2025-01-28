FROM debian:bookworm-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        autoconf \
        libtool \
        pkg-config \
        git \
        ca-certificates \
    && git clone --recurse-submodules -b v1.64.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

WORKDIR /grpc

RUN mkdir -p cmake/build \
    && cd cmake/build \
    && cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr \
      ../.. \
    && make -j 4 \
    && make install

COPY ./ /sigrpc-runtime

WORKDIR /sigrpc-runtime/proto/

RUN make

WORKDIR /sigrpc-runtime/build/

RUN cmake .. \
    && make -j 4

FROM gcr.io/distroless/cc-debian12

COPY --from=builder /sigrpc-runtime/build/sigrpc-runtime /usr/local/bin/sigrpc-runtime

USER 1001
