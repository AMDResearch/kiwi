# FlyDSL invoke example

`flydsl_print_example.py` launches a FlyDSL kernel that invokes a C++ callback
`n` times. The callback is registered on a C++ `kiwi::invoke::Context` in
[`kiwi_print_ext`](../common/kiwi_print_ext.hip.cpp), which does not depend on
the kernel frontend, and prints the arguments it receives. See the
[FlyDSL invoke guide](../../docs/flydsl.md).

```bash
cmake -S . -B build/flydsl -DKIWI_BUILD_FLYDSL=ON -DKIWI_BUILD_EXAMPLES=ON \
  -DKIWI_BUILD_INVOKE_SDMA_PING_PONG=OFF -DHIP_ARCHITECTURES=gfx942
cmake --build build/flydsl -j
cd build/flydsl/examples/flydsl
PYTHONPATH=../../python python3 flydsl_print_example.py -n 4
```

Expected output:

```text
[C++ callback] step=0 value=0.00 tag=0xc0ffee00
[C++ callback] step=1 value=0.50 tag=0xc0ffee01
[C++ callback] step=2 value=1.00 tag=0xc0ffee02
[C++ callback] step=3 value=1.50 tag=0xc0ffee03
FlyDSL kernel invoked the C++ callback 4 times
```
