## NEON: ARM's implementation of SIMD instructions

### Registers: SIMD (NEON)

| Name |  Width  |                    Description                   |
|------|---------|--------------------------------------------------|
|  Vn  | 128-bit | Full NEON vector register                        |
|  Qn  | 128-bit | Historical name (Aarch32 carryover) - Same as Vn |
|  Dn  | 64-bit  | Lower 64-bits of Vn                              |
|  Sn  | 32-bit  | Lower 32-bits (used for scalar FP) of Vn         |
|  Hn  | 16-bit  | Lower 16-bit of Vn                               |
|  Bn  | 8-bit   | Lower 8 bits of Vn                               |

<br>

### Data Types
NEON supports multiple interpretation modes for the same binary contents:

- Signed integers: S8, S16, S32, S64
- Unsigned integers: U8, U16, U32, U64
- Floating-point: F16, F32, F64 (Apple enables F64, some ARM designs omit)
- Polynomial (bitfield) arithmetic: P8, P16

<br>

### Register state and ABI rules
| Registers |                 Classification                | Use Freely? |
|-----------|-----------------------------------------------|-------------|
| V0-v7     | Argument/return value registers               | Yes         |
| V8 - V15  | Callee-saved (Must be preserved across calls) | Restore     |
| V16 - V31 | Temporary (caller-saved)                      | Yes         |


### Vector arrangement specifier/suffixes
| Suffix |          Meaning          | Total Lanes |        Lane Size      |
|--------|---------------------------|-------------|-----------------------|
|  .16b  | 16-bytes                  |  16 lanes   | 8-bit (16 x 8 = 128)  |
|  .8h   | 8 halfwords               |  8 lanes    | 16-bit (8 x 16 = 128) |
|  .4s   | 4 single-precision floats |  4 lanes    | 32-bit (32 x 4 = 128) |
|  .2d   | 2 double-precision floats |  2 lanes    | 64-bit (64 x 2 = 128) |
