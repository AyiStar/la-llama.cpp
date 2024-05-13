# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430
SRC_DIR = ./src
TEST_DIR = ./tests
LAMM_SRC = $(SRC_DIR)/loongarch_matmul.cpp

LA_LLAMA_FLAGS = -DLA_LLAMA

.PHONY: mm_bench

mm_bench:
	g++ -c -Wall $(SRC_DIR)/loongarch_matmul.cpp -o $(SRC_DIR)/loongarch_matmul.o -I$(abs_path LLAMA_CPP_DIR)
	$(MAKE) -B -C $(LLAMA_CPP_DIR) benchmark-matmult CPPFLAGS=$(LA_LLAMA_FLAGS) -j4


.PHONY: test
