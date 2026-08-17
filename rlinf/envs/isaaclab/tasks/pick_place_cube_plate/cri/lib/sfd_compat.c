/* Shims for Ubuntu 24.04-built Safetics libs on Ubuntu 22.04 (glibc 2.35). */
#define _GNU_SOURCE
#include <stdlib.h>

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
  return strtol(nptr, endptr, base);
}

/* GCC 13 libstdc++ iostreams init; older libstdc++ has no this symbol. */
void _ZSt21ios_base_library_initv(void) {}
