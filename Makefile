CC = nvcc
FLAGS = -m64 -std=c++17 -lsentencepiece

TARGETS = infer_fp infer_mixed

all: $(TARGETS)

infer_fp: src/infer_fp.cu
	$(CC) $(FLAGS) -o $@ $^

infer_mixed: src/infer_mixed.cu
	$(CC) $(FLAGS) -o $@ $^

clean:
	rm -f $(TARGETS)