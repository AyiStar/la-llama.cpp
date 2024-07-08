# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430
SRC_DIR = ./src
TEST_DIR = ./test

ifdef LAMM_DEBUG
export LLAMA_DEBUG
LAMM_FLAGS += -DLAMM_DEBUG
endif

ifndef NLA_LLAMA
export LLAMA_LOONGARCH = 1
endif

ifdef FORCE
MK_FORCE = -B
endif

ifdef LAMM_OPT_LEVEL
LAMM_FLAGS += -DLAMM_OPT_LEVEL=$(LAMM_OPT_LEVEL)
else
LAMM_FLAGS += -DLAMM_OPT_LEVEL=3
endif


export LAMM_FLAGS

.PHONY: benchmark
benchmark: $(SRC_DIR)/loongarch_matmul.o
	$(MAKE) -C $(LLAMA_CPP_DIR) la-benchmark-matmult $(MK_FORCE) -j8

.PHONY: main
main: $(SRC_DIR)/loongarch_matmul.o
	$(MAKE) -C $(LLAMA_CPP_DIR) main $(MK_FORCE) -j8
	cp $(LLAMA_CPP_DIR)/main $(TEST_DIR)/main

$(SRC_DIR)/loongarch_matmul.o:
	$(MAKE) -C $(LLAMA_CPP_DIR) lamm

.PHONY: clean
clean:
	rm -f $(SRC_DIR)/*.o $(TEST_DIR)/*.o $(TEST_DIR)/la-benchmark-matmult $(TEST_DIR)/main

.PHONY: format
format:
	clang-format -i $(SRC_DIR)/*.h $(SRC_DIR)/*.hpp $(SRC_DIR)/*.cpp test/*.cpp