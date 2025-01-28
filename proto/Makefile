CC	= protoc
CPP_OUT	= ../build
PROTO_PATH	= .
CPP_GRPC_OUT	= ../build
SRCS	= message.proto

all:
	$(CC) --proto_path=$(PROTO_PATH) --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` --grpc_out=$(CPP_GRPC_OUT) $(SRCS)
	$(CC) --proto_path=$(PROTO_PATH) --cpp_out=$(CPP_GRPC_OUT) $(SRCS)

clean:
	$(RM) ../build/*.pb.*
