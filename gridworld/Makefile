CC = g++

OPT_FLAGS = -o3 -std=gnu++11 -Wno-delete-non-virtual-dtor
DEBUG_FLAGS = -g -std=gnu++11 -Wno-delete-non-virtual-dtor

# CFLAGS = ${DEBUG_FLAGS}
CFLAGS = ${OPT_FLAGS}

HEADERS = headers/
PROTO = protos/
EIGEN = eigen3/

SRCDIR:=src
OBJDIR:=obj

SRC := $(shell find $(SRCDIR) -name "[!p]*.cpp")

INCLUDES = -I${HEADERS} -I${EIGEN} -I${PROTO} -I.

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

all: main

main: ${OBJ_FILES} obj/results.pb.o
	${CC} ${INCLUDES} ${CFLAGS} -o main ${OBJ_FILES} obj/results.pb.o -lprotobuf;

proto:
	protoc protos/results.proto --cpp_out=. --python_out=.

obj/%.o: src/%.cpp obj/results.pb.o
	${CC} ${INCLUDES} $(CFLAGS) -c -o $@ $<

obj/results.pb.o: proto
	${CC} ${INCLUDES} ${CFLAGS} -c protos/results.pb.cc -o obj/results.pb.o

clean:
	rm -f main obj/* protos/results.pb.* protos/results_pb2.py
