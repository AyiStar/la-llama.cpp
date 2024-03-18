# Set up the cross compilation and emulation tools for LoongArch


- Use the officially provided [build tools](https://github.com/loongson/build-tools).
- Download the binaries of [cross toolchain](https://github.com/loongson/build-tools/releases/download/2023.08.08/CLFS-loongarch64-8.1-x86_64-cross-tools-gcc-glibc.tar.xz) and [QEMU linux-user](https://github.com/loongson/build-tools/releases/download/2023.08.08/qemu-loongarch64).
- For convenience, set the root dir of build tools and qemu as $LA_TOOLCHAIN and $LA_QEMU, respective. Add $LA_TOOLCHAIN/bin and $LA_QEMU/bin to $PATH.


## Basic Testing

Test with C
```bash
loongarch64-unknown-linux-gnu-gcc hello_loongarch.c -o hello_loongarch
qemu-loongarch64 -L $LA_TOOLCHAIN/target/ -E LD_LIBRARY_PATH=$LA_TOOLCHAIN/target/lib64/:LD_LIBRARY_PATH  hello_loongarch
```

Test with C++
```bash
loongarch64-unknown-linux-gnu-g++ hello_loongarch.cpp -o hello_loongarch
qemu-loongarch64 -L $LA_TOOLCHAIN/target/ -E LD_LIBRARY_PATH=$LA_TOOLCHAIN/loongarch64-unknown-linux-gnu/lib/:LD_LIBRARY_PATH  hello_loongarch
```