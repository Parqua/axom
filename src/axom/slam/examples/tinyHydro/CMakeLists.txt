# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Axom Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#------------------------------------------------------------------------------
# Slam version of tinyHydro
#------------------------------------------------------------------------------

set(tinyHydro_lib_headers
    TinyHydroTypes.hpp
    HydroC.hpp
    Part.hpp
    State.hpp
    VectorXY.hpp
    PolygonMeshXY.hpp
    )

set(tinyHydro_lib_sources
    TinyHydroTypes.cpp
    HydroC.cpp
    Part.cpp
    State.cpp
    PolygonMeshXY.cpp
    )


set(slam_tiny_hydro_lib_depends_on core slam slic fmt)

#------------------------------------------------------------------------------
# Build the tinyHydro library
#------------------------------------------------------------------------------
blt_add_library(
    NAME        slam_tinyHydro_ex
    SOURCES     ${tinyHydro_lib_sources}
                ${tinyHydro_lib_headers}
    DEPENDS_ON  ${slam_tiny_hydro_lib_depends_on}
    FOLDER      axom/slam/examples 
    OBJECT      TRUE )

list(APPEND slam_tiny_hydro_tests_depends_on slam_tinyHydro_ex)

#------------------------------------------------------------------------------
# Add gtest based tests for tinyHydro
#------------------------------------------------------------------------------
if(AXOM_ENABLE_TESTS)
    blt_add_executable(
        NAME        slam_tinyHydro_unitTests_ex
        SOURCES     tests/slam_tinyHydro_unitTests.cpp
        OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
        DEPENDS_ON  ${slam_tiny_hydro_tests_depends_on} gtest
        FOLDER      axom/slam/examples )

    blt_add_test(
        NAME        slam_tinyHydro_unitTests
        COMMAND     slam_tinyHydro_unitTests_ex )
endif()

#------------------------------------------------------------------------------
# Add the standalone executables for the sod example
#------------------------------------------------------------------------------
blt_add_executable(
    NAME        slam_tinyHydro_sod_ex
    SOURCES     tests/slam_tinyHydro_sod1DTwoPart.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  ${slam_tiny_hydro_tests_depends_on} 
    FOLDER      axom/slam/examples )

if(AXOM_ENABLE_TESTS)
    blt_add_test(
        NAME        slam_tinyHydro_sod
        COMMAND     slam_tinyHydro_sod_ex )
endif()

#------------------------------------------------------------------------------
# Add the standalone executables for the sedov example
# Note: Not a test since the runtime is a bit high in debug mode (~40s)
#------------------------------------------------------------------------------
blt_add_executable(
    NAME        slam_tinyHydro_sedov_ex
    SOURCES     tests/slam_tinyHydro_sedovTwoPart.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  ${slam_tiny_hydro_tests_depends_on}
    FOLDER      axom/slam/examples )

