#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check(stmt)                                                                                                    \
    if (!stmt) {                                                                                                       \
        puts("Test " #stmt " failed.");                                                                                \
        return 1;                                                                                                      \
    }
