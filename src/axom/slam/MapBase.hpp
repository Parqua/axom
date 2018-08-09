/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-741217
 *
 * All rights reserved.
 *
 * This file is part of Axom.
 *
 * For details about use and distribution, please read axom/LICENSE.
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * \file MapBase.hpp
 *
 * \brief Contains an Abstract class MapBase
 *
 */

#ifndef SLAM_MAPBASE_HPP_
#define SLAM_MAPBASE_HPP_

#include "axom/core/Macros.hpp"
#include "axom/core/Types.hpp"

#include "axom/slam/Set.hpp"

namespace axom
{
namespace slam
{

/**
 * \class   MapBase
 *
 * \brief   A base class specifying methods that support operations of a Map,
 *          associating value(s) to each element in a Set. MapBase can be used
 *          as a base class pointer to a templated Map object.
 * \see     Map
 *
 */

class MapBase
{
public:
  using SetIndex = Set::IndexType;
  using SetPosition = Set::PositionType;

public:
  //MapBase(){};
  virtual ~MapBase() {};

  /**
   * \brief Get the number of entities in the set used by this map
   * \return The number of entities in the set used in the map.
   */
  virtual SetPosition size() const = 0;

  /**
   * \brief Checks whether the Map is valid.
   * \return   True if valid, false otherwise.
   */
  virtual bool        isValid(bool verboseOutput) const = 0;

private:
  /**
   * \brief Utility function to verify that the given SetPosition is in a valid
   * range.
   */
  virtual void verifyPosition(SetPosition)       const = 0;

};


} // end namespace slam
} // end namespace axom



#endif // SLAM_MAPBASE_HPP_