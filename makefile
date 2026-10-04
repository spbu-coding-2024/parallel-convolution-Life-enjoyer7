CC = gcc
CFLAGS = -std=gnu11 -Wall -g -O2


OPENCV_PREFIX ?= /usr/local/opencv3.4
INCLUDES = -I src -I$(OPENCV_PREFIX)/include

OPENCV_LIBS_TEST = -L$(OPENCV_PREFIX)/lib \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -lstdc++ -lm

OPENCV_LIBS_MAIN = -L$(OPENCV_PREFIX)/lib \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -lstdc++ -lm \
              -Wl,-rpath,$(OPENCV_PREFIX)/lib




MAIN_PATH = src/main.c src/main_utils.c
FILTER_PATH = src/filter.c
TEST_PATH = tests/tests.c tests/tests_utils.c src/main_utils.c
BUILD_DIR = build
NAME = filter
NAME_TEST = tests

OPENCV_LIB_PATH = $(OPENCV_PREFIX)/lib

build: $(BUILD_DIR)/$(NAME)

test: $(BUILD_DIR)/$(NAME_TEST)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME_TEST)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) src/main_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_MAIN)   -o $@

$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) tests/tests_utils.h src/main_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_TEST)   -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test clean