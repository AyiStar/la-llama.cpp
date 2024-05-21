# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430
SRC_DIR = ./src
TEST_DIR = ./tests

ifdef DEBUG
export LLAMA_DEBUG = 1
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

$(SRC_DIR)/loongarch_matmul.o:
	$(MAKE) -C $(LLAMA_CPP_DIR) lamm

.PHONY: mm_bench
mm_bench: $(SRC_DIR)/loongarch_matmul.o
	$(MAKE) -C $(LLAMA_CPP_DIR) la-benchmark-matmult $(MK_FORCE) -j8

.PHONY: main
main: $(SRC_DIR)/loongarch_matmul.o
	$(MAKE) -C $(LLAMA_CPP_DIR) main $(MK_FORCE) -j8
	cp $(LLAMA_CPP_DIR)/main $(SRC_DIR)

.PHONY: clean
clean:
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/la-benchmark-matmult