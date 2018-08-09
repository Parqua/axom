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

#ifndef MINT_MESH_BLUEPRINT_HPP_
#define MINT_MESH_BLUEPRINT_HPP_

// mint includes
#include "axom/mint/config.hpp"      // for compile-time definitions
#include "axom/mint/mesh/CellTypes.hpp"   // for Topology

// C/C++ includes
#include <string>           // for std::string

/*!
 * \file
 *
 * \brief Provides convenience methods for navigating and querying a
 *  Sidre group hierarchy according to the conventions outlined in the
 *  <a href="http://llnl-conduit.readthedocs.io/en/latest/"> mesh blueprint</a>.
 *
 */

namespace axom
{

// Sidre Forward Declarations
namespace sidre
{
class Group;
}

namespace mint
{

// Mint Forward Declarations;
class Extent;

namespace blueprint
{

#ifdef MINT_USE_SIDRE

/*!
 * \brief Checks if the given root group conforms to the mesh blueprint.
 *
 * \param [in] group pointer to the blueprint root group.
 * \return status true if the group conforms to the blueprint, else, false.
 *
 * \note A root group conforms to the computational mesh blueprint if it
 *  satisfies the following conditions:
 *  (a) it has a "coordsets" child group
 *  (b) it has a "topologies" child group
 *  (c) it has a "fields" child group
 *
 * \note This method emits warnings for each missing group it encounters.
 *
 * \pre group != nullptr
 */
bool isValidRootGroup( const sidre::Group* group );

/*!
 * \brief Checks if the given topology group conforms to the blueprint.
 *
 * \param [in] topo pointer to a topology group.
 * \return status true if the group conforms to the blueprint, else, false.
 *
 * \note This routine performs cursory mesh-agnostic checks, i.e., checks that
 *  the necessary child views exists etc., but, it does not validate the content
 *  of those view at this level.
 *
 * \note A valid topology group conforms to the computational mesh blueprint if
 *  it satisfies the following conditions:
 *  (a) it has a "type" view
 *  (b) it has a "coordset" view
 *  (c) the "type" and "coordset" views are both string views.
 *
 * \pre topo != nullptr
 */
bool isValidTopologyGroup( const sidre::Group* topo );

/*!
 * \brief Checks if the given coordset group conforms to the blueprint.
 *
 * \param [in] coordset pointer to a coordset group.
 * \return status true if the group conforms to the blueprint, else, false.
 *
 * \note This routine performs cursory mesh-agnostic checks, i.e., checks that
 *  the necessary child views exists etc., but, it does not validate the content
 *  of those view at this level.
 *
 * \note A valid coordset group conforms to the computational mesh blueprint
 *  if  it satisfies the following conditions:
 *  (a) it has a "type" view
 *  (b) the "type" view is a string view
 *
 * \pre coordset != nullptr
 */
bool isValidCoordsetGroup( const sidre::Group* coordset );

/*!
 * \brief Returns the requested mesh topology group.
 *
 * \param [in] group the blueprint mesh root group.
 * \param [in] topo optional argument to specify the topology name.
 *
 * \note If a topology is not specified with the second argument, the
 *  code returns the first group under the topologies group.
 *
 * \return topo pointer to the requested topology group or nullptr, if
 *  the requested topology group does not exists or is invalid.
 *
 * \pre group != nullptr
 * \post topo != nullptr
 * \post blueprint::isValidTopologyGroup( topo ) == true
 *
 */
const sidre::Group* getTopologyGroup( const sidre::Group* group,
                                      const std::string& topo="" );

/*!
 * \brief Initialize the topology group.
 *
 * \param [in] group the blueprint mesh root group.
 * \param [in] topo the name of the topology to initialize.
 * \param [in] coordset the coordset to associate with the given topology.
 * \param [in] type the type of the given topology.
 *
 * \pre group != nullptr
 */
void initializeTopologyGroup( sidre::Group* group,
                              const std::string& topo,
                              const std::string& coordset,
                              const std::string& type );

/*!
 * \brief Returns the coordset group associated with the given topology group.
 *
 * \param [in] group the mesh blueprint root group.
 * \param [in] topology the topology group associated with the requested
 *  coordset.
 *
 * \return coordset pointer to the coordset associated with given topology,
 *  else false.
 *
 * \pre blueprint::isValidRootGroup( group )
 * \pre blueprint::isValidTopologyGroup( topology )
 * \post blueprint::isValidCoordsetGroup( coordset )
 */
const sidre::Group* getCoordsetGroup( const sidre::Group* group,
                                      const sidre::Group* topology );

/*!
 * \brief Returns the coordset group associated with the given topology group.
 *
 * \param [in] group the mesh blueprint root group.
 * \param [in] coords option argument to specify the coordset name.
 *
 * \return coordset pointer to the coordset associated with given topology,
 *  else false.
 *
 * \pre blueprint::isValidRootGroup( group )
 * \pre blueprint::isValidTopologyGroup( topology )
 * \post blueprint::isValidCoordsetGroup( coordset )
 */
const sidre::Group* getCoordsetGroup( const sidre::Group* group,
                                      const std::string& coords="" );

/*!
 * \brief Returns the mesh type and dimension given a root group that conforms
 *  to the computational mesh blueprint.
 *
 * \param [out] mesh_type the mint mesh type
 * \param [out] dimension the mesh dimension
 * \param [in] group pointer to the root group
 * \param [in] topo optional argument corresponding to the mesh topology.
 *
 * \note If a topology is not specified, the code assumes that the first group
 *  under the 'topologies' group corresponds to the mesh topology.
 *
 * \pre group != nullptr
 *
 * \see MeshTypes
 */
void getMeshTypeAndDimension( int& mesh_type, int& dimension,
                              const sidre::Group* group,
                              const std::string& topology="" );

/*
 * \brief Return the whether the mesh has mixed cell types.
 *
 * \param [in] group pointer to the root group.
 * \param [in] topo optional argument corresponding to the mesh topology.
 *
 * \note If a topology is not specified, the code assumes that the first group
 *  under the 'topologies' group corresponds to the mesh topology.
 */
bool hasMixedCellTypes( const sidre::Group* group, const std::string& topo="" );

/*!
 * \brief Returns the origin, spacing and extent of a uniform mesh from the
 *  corresponding coordset and topology groups associated with the mesh.
 *
 * \param [in] dim the expected dimension of the mesh
 * \param [in] coordset pointer to the associated coordset group
 * \param [in] topology pointer to the associated topology group
 * \param [out] origin  buffer to store the origin of the uniform mesh
 * \param [out] spacing buffer to store the spacing of the uniform mesh
 * \param [out] extent buffer to store the extent of the uniform mesh
 *
 * \note The `origin` and `spacing` pointers must point to buffers that have
 *  at least `dim` entries.
 *
 * \note `extent` must point to a buffer that has at least  \f$ 2 \times dim \f$
 *  entries, such that `extent[ i*2 ]`, `extent[ i*2+1 ]` hold the min and max
 *  values of the extent along the ith dimension respectively.
 *
 * \pre 1 <= dim <= 3
 * \pre isValidCoordsetGroup( coordset )
 * \pre isValidTopologyGroup( topology )
 * \pre origin != nullptr
 * \pre spacing != nullptr
 * \pre extent != nullptr
 *
 * \see setUniformMesh()
 */
void getUniformMesh( int dim,
                     const sidre::Group* coordset,
                     const sidre::Group* topology,
                     double* origin,
                     double* spacing,
                     int64* extent );

/*!
 * \brief Populates the specified Coordset & Topology groups with the metadata
 *  for a uniform mesh
 *
 * \param [in] dim the mesh dimesion
 * \param [in] origin pointer to array with the origin's coordinates
 * \param [in] spacing pointer to array with the spacing along each dimension
 * \param [in] extent pointer to the associated Extent object
 * \param [in] coordset pointer to the coordset group
 * \param [in] topology pointer to the topology group
 *
 * \note The `origin` and `spacing` pointers must point to buffers that have
 *  at least `dim` entries.
 *
 * \pre 1 <= dim <= 3
 * \pre origin != nullptr
 * \pre spacing != nullptr
 * \pre extent != nullptr
 * \pre coordset != nullptr
 * \pre topology != nullptr
 * \pre extent->getDimesion() == dim
 *
 * \post isValidCoordsetGroup( coordset )
 * \post isValidTopologyGroup( topology
 *
 * \see getUniformMesh()
 */
void setUniformMesh( int dim,
                     const double* origin,
                     const double* spacing,
                     const mint::Extent* extent,
                     sidre::Group* coordset,
                     sidre::Group* topology );

/*!
 * \brief Gets the extent of a curvilinear structured mesh.
 *
 * \param [in] dim the dimension of the mesh.
 * \param [in] topology pointer to the topology group of the mesh.
 * \param [out] extent buffer where the mesh extent will be stored.
 *
 * \note `extent` must point to a buffer that has at least  \f$ 2 \times dim \f$
 *  entries, such that `extent[ i*2 ]`, `extent[ i*2+1 ]` hold the min and max
 *  values of the extent along the ith dimension respectively.
 *
 * \pre 1 <= dim <= 3
 * \pre blueprint::isValidTopologyGroup( topology )
 * \pre extent != nullptr
 */
void getCurvilinearMeshExtent( int dim,
                               const sidre::Group* topology,
                               int64* extent );

/*!
 * \brief Sets the extent of a curvilinear structured mesh
 *
 * \param [in] dim the dimension of the mesh.
 * \param [in] extent pointer to an extent object.
 * \param [in] topology pointer to the topology group of the mesh.
 *
 * \pre 1 <= dim <= 3
 * \pre extent != nullptr
 * \pre topology != nullptr
 * \pre extent->getDimension()==dim
 */
void setCurvilinearMeshExtent( int dim,
                               const mint::Extent* extent,
                               sidre::Group* topology );

/*!
 * \brief Gets the extent of a rectilinear mesh
 *
 * \param [in] dim the dimension of the mesh.
 * \param [in] coordset pointer to the coordset group associated with the mesh
 * \param [in] topology pointer to the topology group associated with the mesh
 * \param [out] extent pointer to buffer where to store the mesh extent.
 *
 * \note `extent` must point to a buffer that has at least  \f$ 2 \times dim \f$
 *  entries, such that `extent[ i*2 ]`, `extent[ i*2+1 ]` hold the min and max
 *  values of the extent along the ith dimension respectively.
 *
 *  \pre 1 <= dim <= 3
 *  \pre blueprint::isValidCoordsetGroup( coordset )
 *  \pre blueprint::isValidTopologyGroup( topology )
 *  \pre extent != nullptr
 */
void getRectilinearMeshExtent( int dim,
                               const sidre::Group* coordset,
                               const sidre::Group* topology,
                               int64* extent );

/*!
 * \brief Sets the extent of a rectilinear mesh.
 *
 * \param [in] dim the mesh dimension.
 * \param [in] extent pointer to the extent object associated with the mesh.
 * \param [in] topology pointer to the topology group associated with the mesh.
 *
 * \pre extent != nullptr
 * \pre topology != nullptr
 * \pre extent->getDimension()==dim
 */
void setRectilinearMeshExtent( int dim,
                               const mint::Extent* extent,
                               sidre::Group* topology );

#endif /* MINT_USE_SIDRE */

} /* namespace blueprint */

} /* namespace mint */
} /* namespace axom */

#endif /* MINT_MESH_BLUEPRINT_HPP_ */