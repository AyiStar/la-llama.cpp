# Set up the cross compilation and emulation tools for LoongArch

- Use the officially provided [cross tools](http://www.loongnix.cn/zh/toolchain/GNU/).
- Download the binaries of [cross toolchain](http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.3.tar.xz) and [QEMU linux-user](https://github.com/loongson/build-tools/releases/download/2023.08.08/qemu-loongarch64).
- For convenience, set the root dir of build tools and qemu as $LA_TOOLCHAIN and $LA_QEMU, respective. Add $LA_TOOLCHAIN/bin and $LA_QEMU/bin to $PATH.


## Basic Testing

Test with C
```bash
loongarch64-linux-gnu-gcc hello_loongarch.c -o hello_loongarch
qemu-loongarch64 -L $LA_TOOLCHAIN/loongarch64-linux-gnu/sysroot/usr/ -E LD_LIBRARY_PATH=$LA_TOOLCHAIN/loongarch64-linux-gnu/sysroot/usr/lib64/:LD_LIBRARY_PATH  hello_loongarch
```

Test with C++
```bash
loongarch64-linux-gnu-g++ hello_loongarch.cpp -o hello_loongarch
qemu-loongarch64 -L $LA_TOOLCHAIN/target/ -E LD_LIBRARY_PATH=$LA_TOOLCHAIN/loongarch64-linux-gnu/lib/:LD_LIBRARY_PATH  hello_loongarch
```

Test LASX support
```bash
loongarch64-linux-gnu-gcc test_lasx.c -o test_lasx -mlasx
```