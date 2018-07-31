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

/*!
 * \file
 *
 * \brief Illustrates how to construct and operate on a rectilinear mesh.
 */

// Axom includes
#include "axom/core/Macros.hpp"           // for AXOM_NOT_USED
#include "axom/core/utilities/Utilities.hpp"  // for random_real()

// Mint includes
#include "axom/mint/config.hpp"
#include "axom/mint/mesh/RectilinearMesh.hpp"
#include "axom/mint/utils/vtk_utils.hpp"

// C/C++ includes
#include <cmath>

// aliases
namespace mint      = axom::mint;
namespace utilities = axom::utilities;
using IndexType     = mint::IndexType;

void exponential_distribution( double origin, IndexType N, double* x )
{
  SLIC_ASSERT( x != nullptr );

  constexpr double beta = 0.02;
  const double expbeta  = exp( beta );
  const double invf     = 1 / ( expbeta - 1.0 );

  x[ 0 ] = origin;
  for ( int i=1 ; i < N ; ++i )
  {
    const double prev = x[ i-1 ];
    const double dx   = ( exp( i*beta ) - 1.0 ) * invf;
    x[ i ] = prev + dx;
  }

}

//------------------------------------------------------------------------------
int main ( int AXOM_NOT_USED(argc), char** AXOM_NOT_USED(argv) )
{
  constexpr int N = 100;

  // STEP 0: construct a N x N rectilinear mesh
  mint::RectilinearMesh mesh( N, N );
  double* x = mesh.getCoordinateArray( mint::X_COORDINATE );
  double* y = mesh.getCoordinateArray( mint::Y_COORDINATE );
  exponential_distribution( -5.0, N, x );
  exponential_distribution(  0.0, N, y );

  // STEP 1: add fields
  double* phi = mesh.createField< double >( "phi", mint::NODE_CENTERED );

  // STEP 2: compute a fiducial field
  const IndexType numNodes = mesh.getNumberOfNodes();
  for ( IndexType inode=0 ; inode < numNodes ; ++inode )
  {
    phi[ inode ] = utilities::random_real( -10.0, 10.0 );
  } // END for all nodes

  // STEP 3: dump file
  mint::write_vtk( &mesh, "rectilinear_mesh.vtk" );

  return 0;
}
