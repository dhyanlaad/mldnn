CC := clang
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

COMMON_FLAGS := -O3 -ffast-math -ffp-contract=fast -fno-math-errno -pthread \
	-Wall -Wextra -Wpedantic

ifeq ($(UNAME_S),Darwin)
  ifneq ($(UNAME_M),arm64)
    $(error These NEON kernels require Apple Silicon (arm64))
  endif
  NATIVE_FLAGS := -mcpu=native
  SHARED_FLAGS := -dynamiclib
  LIB_EXT := dylib
else
  $(error The native benchmark kernels currently support macOS on Apple Silicon only)
endif

CFLAGS ?= $(COMMON_FLAGS) $(NATIVE_FLAGS)

.PHONY: all clean check

all: libfast_fem.$(LIB_EXT) libfast_milstein.$(LIB_EXT)

libfast_fem.$(LIB_EXT): fast_fem_all.c
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $< -o $@

libfast_milstein.$(LIB_EXT): fast_milstein.c
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $< -o $@

check: all
	file libfast_fem.$(LIB_EXT) libfast_milstein.$(LIB_EXT)

clean:
	rm -f libfast_fem.dylib libfast_milstein.dylib
