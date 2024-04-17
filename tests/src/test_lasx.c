#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <lasxintrin.h>

int32_t arr[] = {
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16,
    0, 0, 0, 0, 0, 0, 0, 0
};


int main() {
    // Add two fixed point vectors
    __m256i v0 = __lasx_xvldx(arr, 0);
    __m256i v1 = __lasx_xvldx(arr, 32);
    __m256i v2 = __lasx_xvadd_w(v0, v1);
    __lasx_xvstx(v2, arr, 64);
    for (int i = 0; i < 8; i++) {
        if (arr[i + 16] != arr[i] + arr[i + 8]) {
            printf("Sum at index %d is %d, expect %d\n", i, arr[i + 16], arr[i] + arr[i + 8]);
            return -1;
        }
    }
    printf("Success!\n");
    return 0;
}