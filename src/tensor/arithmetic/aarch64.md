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

### NEON Data Types
NEON supports multiple interpretation modes for the same binary contents:

- Signed integers: S8, S16, S32, S64
- Unsigned integers: U8, U16, U32, U64
- Floating-point: F16, F32, F64 (Apple enables F64, some ARM designs omit)
- Polynomial (bitfield) arithmetic: P8, P16

<br>

### NEON: Register state and ABI rules
| Registers |                 Classification                | Use Freely? |
|-----------|-----------------------------------------------|-------------|
| V0-v7     | Argument/return value registers               | Yes         |
| V8 - V15  | Callee-saved (Must be preserved across calls) | Restore     |
| V16 - V31 | Temporary (caller-saved)                      | Yes         |
