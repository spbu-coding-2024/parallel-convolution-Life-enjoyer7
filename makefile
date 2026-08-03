CC = g++
CFLAGS = -Wall -g -O2
INCLUDES = -I src -I/usr/local/opencv3.4/include

OPENCV_LIBS_TEST = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \

OPENCV_LIBS_MAIN = -L/usr/local/opencv3.4/lib \
              -lopencv_highgui \
              -lopencv_imgcodecs \
              -lopencv_imgproc \
              -lopencv_core \
              -Wl,-rpath,/usr/local/opencv3.4/lib

PTHREADFLAGS = -pthread


MAIN_PATH = src/main.c src/main_utils.c
FILTER_PATH = src/filter.c
TEST_PATH = tests/tests.c tests/tests_utils.c
BENCH_PATH = benchmarks/bench.c src/main_utils.c tests/tests_utils.c
BUILD_DIR = build
NAME = filter
NAME_TEST = tests
NAME_BENCH = bench

OPENCV_LIB_PATH = /usr/local/opencv3.4/lib

# Число повторов на каждую точку замера в `make benchmark` (без учёта прогрева).
REPEAT ?= 5
BENCH_OUT_DIR = benchmarks/generated

build: $(BUILD_DIR)/$(NAME)

test: $(BUILD_DIR)/$(NAME_TEST)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && $(BUILD_DIR)/$(NAME_TEST)

bench-bin: $(BUILD_DIR)/$(NAME_BENCH)

clean_benchmark:
	rm -rf $(BENCH_OUT_DIR)

# Полный перебор картинка×фильтр×стратегия -> CSV -> PNG-графики (matplotlib).
benchmark: bench-bin
	mkdir -p $(BENCH_OUT_DIR)
	export LD_LIBRARY_PATH=$(OPENCV_LIB_PATH):$$LD_LIBRARY_PATH && \
		$(BUILD_DIR)/$(NAME_BENCH) $(REPEAT) > $(BENCH_OUT_DIR)/bench_results.csv
	python3 benchmarks/plot.py $(BENCH_OUT_DIR)/bench_results.csv $(BENCH_OUT_DIR)

$(BUILD_DIR)/$(NAME): $(MAIN_PATH) src/main_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(MAIN_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_MAIN) $(PTHREADFLAGS)  -o $@

$(BUILD_DIR)/$(NAME_TEST): $(TEST_PATH) tests/tests_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(TEST_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_TEST) $(PTHREADFLAGS)  -o $@

$(BUILD_DIR)/$(NAME_BENCH): $(BENCH_PATH) src/main_utils.h tests/tests_utils.h $(FILTER_PATH)
	mkdir -p $(BUILD_DIR)
	$(CC) $(BENCH_PATH) $(FILTER_PATH) $(CFLAGS) $(INCLUDES) $(OPENCV_LIBS_TEST) $(PTHREADFLAGS)  -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: build test clean bench-bin benchmark