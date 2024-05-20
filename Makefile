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
endif


export LAMM_FLAGS

.PHONY: mm_bench

mm_bench:
	@echo "Compiling the project with LLAMA_LOONGARCH=$(LLAMA_LOONGARCH), LAMM_OPT_LEVEL=$(LAMM_OPT_LEVEL)" 
	rm -f $(SRC_DIR)/loongarch_matmul.o
	$(MAKE) -C $(LLAMA_CPP_DIR) la-benchmark-matmult $(MK_FORCE) -j8
