UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

COMMON_FLAGS := -O3 -ffp-contract=off -fno-math-errno -pthread \
	-Wall -Wextra -Wpedantic

ifeq ($(UNAME_S),Darwin)
  SHARED_FLAGS := -dynamiclib
  LIB_EXT := dylib
  ifeq ($(UNAME_M),arm64)
    NATIVE_FLAGS := -mcpu=native
  endif
else ifeq ($(UNAME_S),Linux)
  SHARED_FLAGS := -shared -fPIC
  LIB_EXT := so
else
  $(error The native benchmark kernels currently support macOS and Linux)
endif

CFLAGS ?= $(COMMON_FLAGS) $(NATIVE_FLAGS)

.PHONY: all clean check

all: libfast_fem.$(LIB_EXT) libfast_milstein.$(LIB_EXT)

libfast_fem.$(LIB_EXT): fast_fem_all.c
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $< -o $@ -lm

libfast_milstein.$(LIB_EXT): fast_milstein.c
	$(CC) $(CFLAGS) $(SHARED_FLAGS) $< -o $@ -lm

check: all
	file libfast_fem.$(LIB_EXT) libfast_milstein.$(LIB_EXT)

clean:
	rm -f libfast_fem.dylib libfast_milstein.dylib libfast_fem.so libfast_milstein.so
