// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/* WARNING: notice that based on the hexadecimal number in the last line
   in the hash function the resulting floating point value is very likely
   on the order of 0.5. */
__host__ __device__ unsigned int hash_val(unsigned int a, unsigned int seed)
{
    a ^= seed;
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) + (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a ^ 0xd3a2646c) + (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) + (a >> 16);
    return a;
}

/* return 1e-5 for float [sizeof(float)=4] and 1e-12 for double [sizeof(double)=8] types */
template<typename WeightType>
__host__ __device__ WeightType scaling_factor()
{
    return (sizeof(WeightType) == 4) ? 1e-5f : 1e-12;
}
