# Cross compile llama.cpp to RISC-V on Ubuntu

## Install Prerequisites
On Ubuntu 20.04, install the following packages for riscv64 cross compiler:
- riscv64-linux-gnu
- g++-riscv64-linux-gnu

Then we get riscv64-linux-gnu-gcc and riscv64-linux-gnu-g++.

## Cross Compilation
Cross compile llama.cpp with Makefile:
```bash
make clean
make RISCV_CROSS_COMPILE=1 RISCV=1
```

## Run with QEMU-riscv64:
```bash
qemu-riscv64 -L /usr/riscv64-linux-gnu/ -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 ./main -m $LLAMA_GGUF_PATH/llama-2-7b.Q4_0.gguf -n 512 -p "Building a website can be done in 10 simple steps:\nStep 1:" -e -t 1
```