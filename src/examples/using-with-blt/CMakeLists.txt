# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Axom Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Example that shows how to use an installed instance of Axom 
# in a BLT-based build system.
#
# Configuration variables are stored in a CMake cache file 'host-config.cmake'
# which defines paths to blt, axom and possibly other TPLs.
# It also contains information about the compiler used to build axom.
#------------------------------------------------------------------------------
#
# To build:
#  mkdir build
#  cd build
#  cmake -C ../host-config.cmake ..
#  make
#  ./bin/example
#
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)

project(using_with_blt)

# Define some config options
option(EXAMPLE_VERBOSE_OUTPUT "Verbose output about imported targets" OFF)


#------------------------------------------------------------------------------
# Set up BLT with validity checks
#------------------------------------------------------------------------------

# Check that path to BLT is provided and valid
if(NOT DEFINED BLT_SOURCE_DIR OR NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
    message(FATAL_ERROR "Missing required 'BLT_SOURCE_DIR' variable pointing to a valid blt")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)


#------------------------------------------------------------------------------
# Check for AXOM_DIR and use CMake's find_package to import axom's targets
#------------------------------------------------------------------------------
if(NOT DEFINED AXOM_DIR OR NOT EXISTS ${AXOM_DIR}/lib/cmake/axom-config.cmake)
    message(FATAL_ERROR "Missing required 'AXOM_DIR' variable pointing to an installed axom")
endif()

find_package(axom REQUIRED
             NO_DEFAULT_PATH 
             PATHS ${AXOM_DIR}/lib/cmake)

#------------------------------------------------------------------------------
# Set up example target that depends on axom
#------------------------------------------------------------------------------

blt_add_executable(NAME       example 
                   SOURCES    example.cpp
                   DEPENDS_ON axom fmt)


#------------------------------------------------------------------------------
# Optionally, print out information about imported targets
#------------------------------------------------------------------------------
if(EXAMPLE_VERBOSE_OUTPUT)
    foreach(_target axom cli11 fmt sparsehash conduit::conduit example)
        blt_print_target_properties(TARGET ${_target})  
    endforeach()
endif()

