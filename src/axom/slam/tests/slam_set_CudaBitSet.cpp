// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Axom Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


/*
 *  This file tests the BitSet class of slam
 *  Most tests are parameterized by the values in the testSizes() function
 */

#include "axom/config.hpp"
#include "axom/slic.hpp"

#include "axom/slam/BitSet.hpp"

#include "gtest/gtest.h"

namespace slam = axom::slam;

namespace
{
    // Utility function to initialize a bitset of size \a size
    // where every stride^th bit is set, starting with offset \a offset
    // Note: Use only after establishing that the constructor and set(idx)
    //       functions work properly
    AXOM_HOST_DEVICE
    slam::BitSet generateBitset(int size, int stride = 1, int offset = 0)
    {
        using Index = slam::BitSet::Index;

        slam::BitSet bitset(size);
        for (Index i = offset ; i < size ; i += stride)
        {
            bitset.set(i);
        }

        return bitset;
    }

    
}

class SlamBitSet : public ::testing::TestWithParam<int>
{ };


AXOM_CUDA_TEST(SlamBitSet, checkInitEmpty)
{
    const int NBITS = 80;
    SLIC_INFO("Testing bitset construction (" << NBITS << " bits)");

    slam::BitSet bitset(NBITS);

    EXPECT_TRUE(bitset.isValid());
    EXPECT_EQ(NBITS, bitset.size());
    EXPECT_EQ(0, bitset.count());
    #ifdef AXOM_USE_CUDA
    fprintf(stderr, "CUDAON\n");
    #endif
    // Test that each bit is off
    for (int i = 0 ; i < NBITS ; ++i)
    {
        EXPECT_FALSE(bitset.test(i));
    }
}

AXOM_CUDA_TEST(SlamBitSet, setClearFlipAllBits)
{
    const int NBITS = 80;
    SLIC_INFO("Testing bitset set, clear and flip (" << NBITS << " bits)");

    slam::BitSet bitset(NBITS);

    // count should be 0
    EXPECT_EQ(0, bitset.count());
    EXPECT_TRUE(bitset.isValid());

    // clear all bits, when already empty
    bitset.clear();
    EXPECT_EQ(0, bitset.count());
    EXPECT_TRUE(bitset.isValid());

    // set all bits
    bitset.set();
    EXPECT_EQ(NBITS, bitset.count());
    EXPECT_TRUE(bitset.isValid());

    // clear all bits
    bitset.clear();
    EXPECT_EQ(0, bitset.count());
    EXPECT_TRUE(bitset.isValid());

    // toggle all bits
    bitset.flip();
    EXPECT_EQ(NBITS, bitset.count());
    EXPECT_TRUE(bitset.isValid());
}






//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

#if AXOM_DEBUG
    // add this line to avoid a warning in the output about thread safety
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
#endif

    // create & initialize test logger. finalized when exiting main scope
    axom::slic::UnitTestLogger logger;

    result = RUN_ALL_TESTS();

    return result;
}
