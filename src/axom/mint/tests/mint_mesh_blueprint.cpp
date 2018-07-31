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
#include "axom/mint/mesh/blueprint.hpp"

// Mint includes
#include "axom/mint/core/Array.hpp"           // for mint::Array
#include "axom/mint/config.hpp"          // for compile-time definitions
#include "axom/mint/mesh/MeshTypes.hpp"       // for MeshTypes enum
#include "axom/mint/mesh/Extent.hpp"          // for mint::Extent

// gtest includes
#include "gtest/gtest.h"

#ifdef MINT_USE_SIDRE
#include "sidre/sidre.hpp"

namespace mint  = axom::mint;
namespace sidre = axom::sidre;

//------------------------------------------------------------------------------
// HELPER METHODS
//------------------------------------------------------------------------------
namespace
{

void createExplicitCoords( sidre::Group* c1, int dimension,
                           const std::string& type_string )
{
  c1->createView( "type" )->setString( type_string );
  mint::Array< double > x( c1->createView( "values/x" ), 5, 1 );

  if ( dimension > 1 )
  {
    mint::Array< double > y( c1->createView( "values/y" ), 5, 1 );
  }

  if ( dimension > 2 )
  {
    mint::Array< double > z( c1->createView( "values/z" ), 5, 1 );
  }

}

//------------------------------------------------------------------------------
void createParticleMeshOnSidre( sidre::Group* root, int dimension )
{
  EXPECT_TRUE( root != nullptr );
  EXPECT_TRUE( root->getNumGroups()==0 );
  EXPECT_TRUE( root->getNumViews()==0 );
  EXPECT_TRUE( dimension >= 1 );
  EXPECT_TRUE( dimension <= 3 );

  sidre::Group* coordsets  = root->createGroup( "coordsets" );
  sidre::Group* topologies = root->createGroup( "topologies" );
  root->createGroup( "fields" );

  sidre::Group* c1 = coordsets->createGroup( "c1" );
  createExplicitCoords( c1, dimension, "explicit" );

  sidre::Group* t1 = topologies->createGroup( "t1"  );

  t1->createView( "type" )->setString( "points" );
  t1->createView( "coordset" )->setString( "c1" );

  // NOTE: the generated Sidre hierarchy is not complete, it only consists of
  // information sufficient for the tests herein.
}

//------------------------------------------------------------------------------
void createUniformMeshOnSidre( sidre::Group* root, int dimension )
{
  EXPECT_TRUE( root != nullptr );
  EXPECT_TRUE( root->getNumGroups()==0 );
  EXPECT_TRUE( root->getNumViews()==0 );
  EXPECT_TRUE( dimension >= 1 );
  EXPECT_TRUE( dimension <= 3 );

  sidre::Group* coordsets  = root->createGroup( "coordsets" );
  sidre::Group* topologies = root->createGroup( "topologies" );
  root->createGroup( "fields" );

  sidre::Group* c1 = coordsets->createGroup( "c1" );
  c1->createView( "type" )->setString( "uniform" );
  c1->createView( "dims/i" )->setScalar( 10 );
  c1->createView( "origin/x" )->setScalar( 0 );
  c1->createView( "spacing/dx" )->setScalar( 0.5 );

  if ( dimension > 1 )
  {
    c1->createView( "dims/j" )->setScalar( 10 );
    c1->createView( "origin/y" )->setScalar( 0 );
    c1->createView( "spacing/dy" )->setScalar( 0.5 );
  }

  if ( dimension > 2 )
  {
    c1->createView( "dims/k" )->setScalar( 10 );
    c1->createView( "origin/z" )->setScalar( 0 );
    c1->createView( "spacing/dz" )->setScalar( 0.5 );
  }

  sidre::Group* t1 = topologies->createGroup( "t1" );
  t1->createView( "type" )->setString( "uniform" );
  t1->createView( "coordset" )->setString( "c1" );

  // NOTE: the generated Sidre hierarchy is not complete, it only consists of
  // information sufficient for the tests herein.
}

//------------------------------------------------------------------------------
void createRectilinearMeshOnSidre( sidre::Group* root, int dimension )
{
  EXPECT_TRUE( root != nullptr );
  EXPECT_TRUE( root->getNumGroups()==0 );
  EXPECT_TRUE( root->getNumViews()==0 );
  EXPECT_TRUE( dimension >= 1 );
  EXPECT_TRUE( dimension <= 3 );

  sidre::Group* coordsets  = root->createGroup( "coordsets" );
  sidre::Group* topologies = root->createGroup( "topologies" );
  root->createGroup( "fields" );

  sidre::Group* c1 = coordsets->createGroup( "c1" );
  createExplicitCoords( c1, dimension, "rectilinear" );

  sidre::Group* t1 = topologies->createGroup( "t1"  );
  t1->createView( "type" )->setString( "rectilinear" );
  t1->createView( "coordset" )->setString( "c1" );

  // NOTE: the generated Sidre hierarchy is not complete, it only consists of
  // information sufficient for the tests herein.
}

//------------------------------------------------------------------------------
void createStructuredMeshOnSidre( sidre::Group* root, int dimension )
{
  EXPECT_TRUE( root != nullptr );
  EXPECT_TRUE( root->getNumGroups()==0 );
  EXPECT_TRUE( root->getNumViews()==0 );
  EXPECT_TRUE( dimension >= 1 );
  EXPECT_TRUE( dimension <= 3 );

  sidre::Group* coordsets  = root->createGroup( "coordsets" );
  sidre::Group* topologies = root->createGroup( "topologies" );
  root->createGroup( "fields" );

  sidre::Group* c1 = coordsets->createGroup( "c1" );
  createExplicitCoords( c1, dimension, "explicit" );

  sidre::Group* t1 = topologies->createGroup( "t1"  );
  t1->createView( "type" )->setString( "structured" );
  t1->createView( "coordset" )->setString( "c1" );

  // NOTE: the generated Sidre hierarchy is not complete, it only consists of
  // information sufficient for the tests herein.
}

//------------------------------------------------------------------------------
void createUnstructuredMeshOnSidre( sidre::Group* root, int dimension )
{
  EXPECT_TRUE( root != nullptr );
  EXPECT_TRUE( root->getNumGroups()==0 );
  EXPECT_TRUE( root->getNumViews()==0 );
  EXPECT_TRUE( dimension >= 1 );
  EXPECT_TRUE( dimension <= 3 );

  sidre::Group* coordsets  = root->createGroup( "coordsets" );
  sidre::Group* topologies = root->createGroup( "topologies" );
  root->createGroup( "fields" );

  sidre::Group* c1 = coordsets->createGroup( "c1" );
  createExplicitCoords( c1, dimension, "explicit" );

  sidre::Group* t1 = topologies->createGroup( "t1" );
  t1->createView( "type" )->setString( "unstructured" );
  t1->createView( "coordset" )->setString( "c1" );
  t1->createView( "elements/shape" )->setString( "hex" );
  // NOTE: the generated Sidre hierarchy is not complete, it only consists of
  // information sufficient for the tests herein.
}

}

//------------------------------------------------------------------------------
// UNIT TESTS
//------------------------------------------------------------------------------

TEST( mint_mesh_blueprint, check_valid_root_group )
{
  sidre::DataStore ds;
  sidre::Group* root = ds.getRoot( );

  // an empty group is an invalid blueprint root group
  EXPECT_FALSE( mint::blueprint::isValidRootGroup( root ) );

  // populate the group to make it a valid blueprint root group
  root->createGroup( "coordsets" );
  root->createGroup( "topologies" );
  root->createGroup( "fields" );

  EXPECT_TRUE( mint::blueprint::isValidRootGroup( root ) );

  // create an invalid root group
  root->destroyGroups( );
  EXPECT_FALSE( mint::blueprint::isValidRootGroup( root ) );

  root->createGroup( "foo" );
  root->createGroup( "bar" );
  EXPECT_FALSE( mint::blueprint::isValidRootGroup( root ) );
}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, check_valid_topology_group )
{
  sidre::DataStore ds;
  sidre::Group* root = ds.getRoot( );

  // an empty group is an invalid blueprint group
  EXPECT_FALSE( mint::blueprint::isValidTopologyGroup( root ) );

  // populate the topology group
  root->createView("type")->setString("uniform");
  root->createView("coordset")->setString( "c1" );
  EXPECT_TRUE( mint::blueprint::isValidTopologyGroup( root ) );

  // create an invalid root group
  root->destroyViews();
  root->createView( "foo" )->setString( "bar" );
  EXPECT_FALSE( mint::blueprint::isValidTopologyGroup( root ) );
}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, check_valid_coordset_group )
{
  sidre::DataStore ds;
  sidre::Group* root = ds.getRoot();

  // an empty group is an invalid coordset group
  EXPECT_FALSE( mint::blueprint::isValidCoordsetGroup( root ) );

  // populate the coordset group
  root->createView( "type" )->setString( "explicit" );
  EXPECT_TRUE( mint::blueprint::isValidCoordsetGroup( root ) );
}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, get_topology_and_coordset_group )
{
  sidre::DataStore ds;
  sidre::Group* root = ds.getRoot();

  createUnstructuredMeshOnSidre( root, 3 );
  const sidre::Group* topology =
    mint::blueprint::getTopologyGroup( root );
  EXPECT_TRUE( mint::blueprint::isValidTopologyGroup( topology ) );
  EXPECT_EQ( topology->getName(), "t1" );

  const sidre::Group* coordset =
    mint::blueprint::getCoordsetGroup( root, topology );
  EXPECT_TRUE( mint::blueprint::isValidCoordsetGroup( coordset ) );
  EXPECT_EQ( coordset->getName(), "c1");
  std::string coordset_name (
    const_cast< sidre::View* >( topology->getView("coordset") )->getString() );
  EXPECT_EQ( coordset_name, "c1" );
}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, get_mesh_type_and_dimension )
{
  for ( int dim=1 ; dim <= 3 ; ++dim )
  {
    sidre::DataStore ds;
    sidre::Group* root = ds.getRoot();
    sidre::Group* particle_mesh_group     = root->createGroup("points");
    sidre::Group* uniform_mesh_group      = root->createGroup("uniform");
    sidre::Group* structured_mesh_group   = root->createGroup("structured");
    sidre::Group* rectilinear_mesh_group  = root->createGroup("rectilinear");
    sidre::Group* unstructured_mesh_group = root->createGroup("unstructured");

    // STEP 0: Check particle mesh
    int mesh_type = mint::UNDEFINED_MESH;
    int dimension = -1;
    createParticleMeshOnSidre( particle_mesh_group, dim );
    mint::blueprint::getMeshTypeAndDimension( mesh_type, dimension,
                                              particle_mesh_group );
    EXPECT_EQ( mesh_type, mint::PARTICLE_MESH );
    EXPECT_EQ( dimension, dim );

    // STEP 1: check uniform mesh
    mesh_type = mint::UNDEFINED_MESH;
    dimension = -1;
    createUniformMeshOnSidre( uniform_mesh_group, dim );
    mint::blueprint::getMeshTypeAndDimension( mesh_type, dimension,
                                              uniform_mesh_group );
    EXPECT_EQ( mesh_type, mint::STRUCTURED_UNIFORM_MESH );
    EXPECT_EQ( dimension, dim );

    // STEP 2: check structured mesh
    mesh_type = mint::UNDEFINED_MESH;
    dimension = -1;
    createStructuredMeshOnSidre( structured_mesh_group, dim );
    mint::blueprint::getMeshTypeAndDimension( mesh_type, dimension,
                                              structured_mesh_group );
    EXPECT_EQ( mesh_type, mint::STRUCTURED_CURVILINEAR_MESH );
    EXPECT_EQ( dimension, dim );

    // STEP 3: check rectilinear mesh
    mesh_type = mint::UNDEFINED_MESH;
    dimension = -1;
    createRectilinearMeshOnSidre( rectilinear_mesh_group, dim );
    mint::blueprint::getMeshTypeAndDimension( mesh_type, dimension,
                                              rectilinear_mesh_group );
    EXPECT_EQ( mesh_type, mint::STRUCTURED_RECTILINEAR_MESH );
    EXPECT_EQ( dimension, dim );

    // STEP 4: check unstructured mesh
    mesh_type = mint::UNDEFINED_MESH;
    dimension = -1;
    createUnstructuredMeshOnSidre( unstructured_mesh_group, dim );
    mint::blueprint::getMeshTypeAndDimension( mesh_type, dimension,
                                              unstructured_mesh_group );
    EXPECT_EQ( mesh_type, mint::UNSTRUCTURED_MESH );
    EXPECT_EQ( dimension, dim );

  } // END for all dimensions
}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, get_set_uniform_mesh )
{
  constexpr int NDIMS        = 3;
  constexpr double MAGIC_VAL = 42.0;
  constexpr double DX        = 0.5;
  const double X0[]          = { MAGIC_VAL, MAGIC_VAL, MAGIC_VAL };
  const double H[]           = { DX, DX, DX };
  const mint::int64 EXTENT[] = { -2,2, -2,2, -2,2 };

  for ( int idim=1 ; idim <= NDIMS ; ++idim )
  {
    sidre::DataStore ds;
    sidre::Group* root = ds.getRoot();
    sidre::Group* coordset = root->createGroup( "coordsets/c1" );
    sidre::Group* topo     = root->createGroup( "topologies/t1" );
    mint::blueprint::initializeTopologyGroup( root, "t1", "c1", "uniform" );

    mint::Extent mesh_extent( idim, EXTENT );

    mint::blueprint::setUniformMesh( idim,X0,H,&mesh_extent,coordset,topo );

    double* origin      = new double[ idim ];
    double* h           = new double[ idim ];
    mint::int64* extent = new mint::int64[ idim*2 ];

    mint::blueprint::getUniformMesh( idim, coordset, topo, origin, h, extent);

    for ( int i=0 ; i < idim ; ++i )
    {
      EXPECT_DOUBLE_EQ( origin[ i ], X0[ i ] );
      EXPECT_DOUBLE_EQ( h[ i ], H[ i ] );

      const int offset = i*2;
      EXPECT_EQ( extent[ offset ], EXTENT[ offset ] );
      EXPECT_EQ( extent[ offset+1 ], EXTENT[ offset+1 ] );
    }

    delete [] origin;
    delete [] h;
    delete [] extent;
  }

}

//------------------------------------------------------------------------------
TEST( mint_mesh_blueprint, get_set_curvilinear_mesh_extent )
{
  constexpr int NDIMS        = 3;
  const mint::int64 EXTENT[] = { -2,2, -2,2, -2,2 };

  for ( int idim=1 ; idim <= NDIMS ; ++idim )
  {
    sidre::DataStore ds;
    sidre::Group* root     = ds.getRoot();
    root->createGroup( "coordsets/c1" );

    sidre::Group* topo     = root->createGroup( "topologies/t1" );
    mint::blueprint::initializeTopologyGroup( root, "t1", "c1", "structured" );

    mint::Extent mesh_extent( idim, EXTENT );
    mint::blueprint::setCurvilinearMeshExtent( idim, &mesh_extent, topo );

    mint::int64* extent = new mint::int64[ idim*2 ];
    mint::blueprint::getCurvilinearMeshExtent( idim, topo, extent );

    for ( int i=0 ; i < idim ; ++i )
    {
      const int offset = i*2;
      EXPECT_EQ( extent[ offset ], EXTENT[ offset ] );
      EXPECT_EQ( extent[ offset+1 ], EXTENT[ offset+1 ] );
    }

    delete [] extent;
  } // END for all dimensions

}

#endif /* MINT_USE_SIDRE */

//------------------------------------------------------------------------------
#include "axom/slic/core/UnitTestLogger.hpp"
using axom::slic::UnitTestLogger;

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  UnitTestLogger logger;  // create & initialize test logger,

  // finalized when exiting main scope

  result = RUN_ALL_TESTS();

  return result;
}
