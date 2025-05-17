#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

void *malloc_aligned(size_t size, size_t align) {
    void *p;
    if (posix_memalign(&p, align, size) != 0) {
        perror("posix_memalign failed");
        exit(1);
    }
    return p;
}