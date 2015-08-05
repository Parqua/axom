/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and further
 * review from Lawrence Livermore National Laboratory.
 */


/*
 * \file testNullSet.cpp
 *
 * Unit tests for the NullSet class
 */

#include "gtest/gtest.h"

#include "meshapi/Set.hpp"
#include "meshapi/NullSet.hpp"

TEST(gtest_meshapi_set,construct_nullset)
{
  asctoolkit::meshapi::Set* s = new asctoolkit::meshapi::NullSet();

  EXPECT_TRUE(s->empty());

  delete s;

  EXPECT_TRUE( true );
}

TEST(gtest_meshapi_set,subscript_fails_nullset)
{
  std::cout << "\n****** Testing subscript access on NullSet -- code is expected to assert and die." << std::endl;

  typedef asctoolkit::meshapi::Set::PositionType SetPosition;
  asctoolkit::meshapi::NullSet n;

  EXPECT_EQ(n.size(), SetPosition()) << "size of null set is defined to be zero";

#ifdef ATK_DEBUG
  // NOTE: ATK_ASSSERT is disabled in release mode, so this test will only fail in debug mode

  // add this line to avoid a warning in the output about thread safety
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_DEATH(n[0],"") << "subscript operator on null set asserts";
#else
  std::cout << "Did not check for assertion failure since assertions are compiled out in release mode." << std::endl;
#endif
}
