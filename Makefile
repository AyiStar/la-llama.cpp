# A wrapper for project-specific building

LLAMA_CPP_DIR = ./llama.cpp-b2430

LA_LLAMA_FLAGS = -DLA_LLAMA

# Tiling
ifdef TILE_BLOCK_SIZE
LA_LLAMA_FLAGS += -DTILE_BLOCK_SIZE=$(TILE_BLOCK_SIZE)
$(info LA_LLAMA: TILE_BLOCK_SIZE is set to $(TILE_BLOCK_SIZE))
else
$(info LA_LLAMA: TILE_BLOCK_SIZE is default to 1)
endif


.PHONY: mm_bench

mm_bench:
	$(MAKE) -B -C $(LLAMA_CPP_DIR) benchmark-matmult CPPFLAGS=$(LA_LLAMA_FLAGS) -j4
