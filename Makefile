# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430
SRC_DIR = ./src
TEST_DIR = ./tests

LA_LLAMA_FLAGS = -DLA_LLAMA

.PHONY: mm_bench

mm_bench:
	$(MAKE) -B -C $(LLAMA_CPP_DIR) benchmark-matmult CPPFLAGS=$(LA_LLAMA_FLAGS) -j4


.PHONY: test
