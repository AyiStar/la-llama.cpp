# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430
SRC_DIR = ./src
TEST_DIR = ./tests

ifdef DEBUG
DEBUG_FLAG = "LLAMA_DEBUG=1"
endif

.PHONY: mm_bench

mm_bench:
	@echo $(LA_LLAMA_FLAGS) 
	$(MAKE) -B -C $(LLAMA_CPP_DIR) la-benchmark-matmult LLAMA_LOONGARCH=1 $(DEBUG_FLAG) -j8


.PHONY: test
