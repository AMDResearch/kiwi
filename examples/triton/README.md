# Triton invoke example

`triton_print_example.py` launches a Triton kernel that invokes a C++ callback
`n` times. The callback is registered on a C++ `kiwi::invoke::Context` in
[`kiwi_print_ext`](../common/kiwi_print_ext.hip.cpp), which does not depend on
the kernel frontend, and prints the arguments it receives. See the
[Triton frontend guide](../../python/README.md#kiwi-invoke-from-triton).

From the repository root:

```bash
cmake -S . -B build -DKIWI_BUILD_TRITON=ON -DKIWI_BUILD_EXAMPLES=ON \
  -DKIWI_BUILD_INVOKE_SDMA_PING_PONG=OFF -DHIP_ARCHITECTURES=gfx942
cmake --build build -j
PYTHONPATH="$PWD/build/python" python3 build/examples/triton/triton_print_example.py -n 4
```

Expected output:

```text
[C++ callback] step=0 value=0.00 tag=0xc0ffee00
[C++ callback] step=1 value=0.50 tag=0xc0ffee01
[C++ callback] step=2 value=1.00 tag=0xc0ffee02
[C++ callback] step=3 value=1.50 tag=0xc0ffee03
Triton kernel invoked the C++ callback 4 times
```
