// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Axom Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#ifndef SPIN_IMPLICIT_GRID_ACCEL__HPP_
#define SPIN_IMPLICIT_GRID_ACCEL__HPP_

#include "RAJA/RAJA.hpp"

#include "axom/config.hpp"
#include "axom/core.hpp"  // for clamp functions
#include "axom/slic.hpp"
#include "axom/slam.hpp"

#include "axom/core/utilities/AnnotationMacros.hpp" //for annotations
#include "axom/core/execution/execution_space.hpp" // for execution spaces for RAJA

#include "axom/primal/geometry/BoundingBox.hpp"
#include "axom/primal/geometry/Point.hpp"
#include "axom/primal/geometry/Vector.hpp"
#include "axom/spin/RectangularLattice.hpp"

#include <vector>

#ifdef AXOM_USE_CUDA
#define LAMBDA_FILLER AXOM_HOST_DEVICE
#else
#define LAMBDA_FILLER
#endif

namespace axom
{
namespace spin
{


/*!
 * \class ImplicitGrid
 *
 * \brief An implicit grid is an occupancy-based spatial index over an indexed
 * set of objects in space.
 *
 * An ImplicitGrid divides a given rectilinear slab of space (defined by an
 * axis aligned bounding box) into uniformly sized cells
 * of a specified resolution.
 * The GridCells of the ImplicitGrid index a subset of the items from an indexed
 * set whose cardinality is specified during the ImplicitGrid's initialization.
 * Users can insert items from the indexed set into an ImplicitGrid by providing
 * the item's bounding box and index.
 *
 * In contrast to a spin::UniformGrid, which encodes an array of indices
 * for each cell in the underlying multidimensional grid,
 * the ImplicitGrid encodes a single array of bins per dimension, each of which
 * has a bitset over the index space.  Thus, the storage overhead of an
 * ImplicitGrid is fixed at initialization time to
 *   \f$ numElts * sum_i { res[i] } \f$ bits.
 * Queries are implemented in terms of unions and intersections of bitsets.
 *
 * One might prefer an ImplicitGrid over a UniformGrid when one expects
 * a relatively dense index relative to the grid resolution (i.e. that
 * there will be many items indexed per bucket).  The ImplicitGrid
 * is designed for quick indexing and searching over a static (and relatively
 * small index space) in a relatively coarse grid.
 */
template<int NDIMS, typename TheIndexType = int, typename ExecSpace=axom::SEQ_EXEC>
class ImplicitGrid_Accel
{
public:
  using IndexType = TheIndexType;
  using GridCell = primal::Point<IndexType, NDIMS>;
  using SpacePoint = primal::Point<double, NDIMS>;
  using SpaceVec = primal::Vector<double, NDIMS>;

  using SpatialBoundingBox = primal::BoundingBox<double, NDIMS>;
  using LatticeType = RectangularLattice<NDIMS, double, IndexType>;

  using SizePolicy = slam::policies::RuntimeSize<IndexType>;
  using ElementSet = slam::OrderedSet<IndexType,IndexType,SizePolicy>;
  using BinSet = slam::OrderedSet<IndexType,IndexType, SizePolicy>;


  using BitsetType = slam::BitSet;
  using BinBitMap = slam::Map<slam::Set<>, BitsetType>;

  /*!
   * \brief Default constructor for an ImplicitGrid
   *
   * \note Users must call initialize() to initialize the ImplicitGrid
   *       after constructing with the default constructor
   */
  ImplicitGrid_Accel() : m_initialized(false) {}

  /*!
   * \brief Constructor for an implicit grid from a bounding box,
   * a grid resolution a number of elements.
   *
   * \param [in] boundingBox Bounding box of domain to index
   * \param [in] gridRes Pointer to resolution for lattice covering bounding box
   * \param [in] numElts The number of elements to be indexed
   *
   * \pre \a gridRes is either NULL or has \a NDIMS coordinates
   * \sa initialize() for details on setting grid resolution
   * when \a gridRes is NULL
   */
  ImplicitGrid_Accel(const SpatialBoundingBox& boundingBox,
               const GridCell* gridRes,
               int numElts)
    : m_bb(boundingBox), m_initialized(false)
  {
    initialize(m_bb, gridRes, numElts);
  }

  /*!
   * \brief Constructor for an implicit grid from arrays of primitive types
   *
   * \param [in] bbMin Lower bounds of mesh bounding box
   * \param [in] bbMax Upper bounds of mesh bounding box
   * \param [in] gridRes Resolution for lattice covering mesh bounding box
   * \param [in] numElts The number of elements in the index space
   *
   * \pre \a bbMin and \a bbMax are not NULL and have \a NDIMS coordinates
   * \pre \a gridRes is either NULL or has \a NDIMS coordinates
   * \sa initialize() for details on setting grid resolution
   * when \a gridRes is NULL
   */
  ImplicitGrid_Accel(const double* bbMin,
               const double* bbMax,
               const int* gridRes,
               int numElts)
    : m_initialized(false)
  {
    SLIC_ASSERT( bbMin != nullptr);
    SLIC_ASSERT( bbMax != nullptr);

    // Set up the grid resolution from the gridRes array
    //   if NULL, GridCell parameter to initialize should also be NULL
    const GridCell* pRes =
      (gridRes != nullptr) ? &m_gridRes : nullptr;

    initialize(
      SpatialBoundingBox( SpacePoint(bbMin), SpacePoint(bbMax) ),
      pRes, numElts);
  }

  /*! Predicate to check if the ImplicitGrid has been initialized */
  bool isInitialized() const { return m_initialized; }


  /*!
   * \brief Initializes an implicit grid or resolution gridRes over an axis
   * aligned domain covered by boundingBox. The implicit grid indexes a set
   * with numElts elements.
   *
   * \param [in] boundingBox Bounding box of domain to index
   * \param [in] gridRes Resolution for lattice covering bounding box
   * \param [in] numElts The number of elements to be indexed
   * \pre The ImplicitGrid has not already been initialized
   *
   * \note When \a gridRes is NULL, we use a heuristic to set the grid
   * resolution to the N^th root of \a numElts. We also ensure that
   * the resolution along any dimension is at least one.
   *
   */
  void initialize(const SpatialBoundingBox& boundingBox,
                  const GridCell* gridRes,
                  int numElts)
  {
    SLIC_ASSERT( !m_initialized);
    AXOM_PERF_MARK_FUNCTION("implicitgrid_init");
    int element_count;
    // Setup Grid Resolution, dealing with possible null pointer
    if(gridRes == nullptr)
    {
      // Heuristic: use the n^th root of the number of elements
      // add 0.5 to round to nearest integer
      double nthRoot = std::pow(static_cast<double>(numElts), 1./ NDIMS );
      int dimRes = static_cast<int>(nthRoot + 0.5);
      m_gridRes = GridCell(dimRes);
    }
    else
    {
      m_gridRes = GridCell(*gridRes);
    }
    // ensure that resolution in each dimension is at least one
    //for(int i=0 ; i< NDIMS ; ++i)
    for_all< ExecSpace >(NDIMS, AXOM_LAMBDA(IndexType i)
    {
      
      m_gridRes[i] = axom::utilities::max( m_gridRes[i], 1);
    });

    // Setup lattice
    m_bb = boundingBox;
    m_lattice = spin::rectangular_lattice_from_bounding_box(boundingBox,
                                                            m_gridRes.array());
    m_elementSet = ElementSet(numElts);
    element_count = m_elementSet.size();
    //for(int i=0 ; i<NDIMS ; ++i)
    for_all< ExecSpace >(NDIMS, AXOM_LAMBDA(IndexType i)
    {
      m_bins[i] = BinSet(m_gridRes[i]);
      m_binData[i] = BinBitMap(&m_bins[i], BitsetType(element_count));
    });

    // Set the expansion factor for each element to a small fraction of the
    // grid's bounding boxes diameter
    // TODO: Add a constructor that allows users to set the expansion factor
    const double EPS = 1e-8;
    m_expansionFactor = m_bb.range().norm() * EPS;

    m_initialized = true;
  }

  /*! Accessor for ImplicitGrid's resolution */
  const GridCell& gridResolution() const { return m_gridRes; }

  /*! Returns the number of elements in the ImplicitGrid's index set */
  int numIndexElements() const { return m_elementSet.size(); }

  /*!
   * \brief Inserts an element with index \a idx and bounding box \a bbox
   * into the implicit grid
   *
   * \param [in] bbox The bounding box of the element
   * \param [in] idx  The index of the element
   *
   * \note \a bbox is intentionally passed by value since insert()
   * modifies its bounds
   */
  void insert(SpatialBoundingBox bbox, IndexType idx)
  {
    SLIC_ASSERT(m_initialized);
    AXOM_PERF_MARK_FUNCTION("implicitgrid_insert");
    // Note: We slightly inflate the bbox of the objects.
    //       This effectively ensures that objects on grid boundaries are added
    //       in all nearby grid cells.

    bbox.expand(m_expansionFactor);

    const GridCell lowerCell = m_lattice.gridCell( bbox.getMin() );
    const GridCell upperCell = m_lattice.gridCell( bbox.getMax() );

    for(int i=0 ; i< NDIMS ; ++i)
    {
      //BinBitMap& binData = m_binData[i];

      const IndexType lower =
        axom::utilities::clampLower(lowerCell[i], IndexType() );
      const IndexType upper =
        axom::utilities::clampUpper(upperCell[i], highestBin(i) );

      for_all< ExecSpace >(lower, upper+1, [=] LAMBDA_FILLER(IndexType j)
      {
        m_binData[i][j].set(idx);
      });
    }
  }

  /*!
   * Finds the candidate elements in the vicinity of query point \a pt
   *
   * \param [in] pt The query point
   * \return A bitset \a bSet whose bits correspond to
   * the elements of the IndexSet.
   * The bits of \a bSet are set if their corresponding element bounding boxes
   * overlap the grid cell containing \a pt.
   */
  BitsetType getCandidates(const SpacePoint& pt) const
  {
    AXOM_PERF_MARK_FUNCTION("implicitgrid_get_candidates_point");
    if(!m_initialized || !m_bb.contains(pt) )
      return BitsetType(0);

    const GridCell gridCell = m_lattice.gridCell(pt);

    // Note: Need to clamp the upper range of the gridCell
    //       to handle points on the upper boundaries of the bbox
    //       This is valid since we've already ensured that pt is in the bbox.
    IndexType idx = axom::utilities::clampUpper(gridCell[0], highestBin(0));
    BitsetType res = m_binData[0][ idx ];
    
    BitsetType identity(res.size());
    identity.flip();
	
    using reduce_pol = typename axom::execution_space< ExecSpace >::reduce_policy;
    RAJA::ReduceBitAnd< reduce_pol, BitsetType > bit_string(res, identity);
   
    //for(int i=1 ; i< NDIMS ; ++i)
    for_all< ExecSpace >(1, NDIMS, AXOM_LAMBDA(IndexType i)
    {
      IndexType loop_idx = axom::utilities::clampUpper(gridCell[i], highestBin(i));
      bit_string &= m_binData[i][loop_idx];
    });
    res = bit_string.get();

    return res;
  }

  /*!
   * Finds the candidate elements in the vicinity of query box \a box
   *
   * \param [in] box The query box
   * \return A bitset \a bSet whose bits correspond to
   * the elements of the IndexSet.
   * The bits of \a bSet are set if their corresponding element bounding boxes
   * overlap the grid cell containing \a box
   */
  BitsetType getCandidates(const SpatialBoundingBox& box) const
  {
    AXOM_PERF_MARK_FUNCTION("implicitgrid_get_candidates_box");
    if(!m_initialized || !m_bb.intersectsWith(box) )
    
      return BitsetType(0);

    const GridCell lowerCell = m_lattice.gridCell(box.getMin());
    const GridCell upperCell = m_lattice.gridCell(box.getMax());

    BitsetType bits = getBitsInRange(0, lowerCell[0], upperCell[0]);
    BitsetType identity(bits.size());
    identity.flip();
	
    using reduce_pol = typename axom::execution_space< ExecSpace >::reduce_policy;
    RAJA::ReduceBitAnd< reduce_pol, BitsetType > bit_string(bits, identity);
   
    
    for_all< ExecSpace >(1, NDIMS, AXOM_LAMBDA(IndexType dim)
    //for(int dim=1 ; dim< NDIMS ; ++dim)
    {
      bit_string &= getBitsInRange(dim, lowerCell[dim], upperCell[dim]);
    });
    bits = bit_string.get();
    return bits;
  }

  /*!
   * Returns an explicit list of candidates in the vicinity of a query object
   *
   * \tparam QueryGeom The type of the query object (e.g. point or box)
   * \param [in] query The query object
   * \return A list of indexes from the IndexSet whose corresponding
   * bounding boxes overlap the grid cell containing \a query
   *
   * \pre This function is implemented in terms of
   * ImplicitGrid::getCandidates(const QueryGeom& ). An overload for the actual
   * \a QueryGeom type (e.g. \a SpacePoint or \a SpatialBoundingBox) must exist.
   *
   * \note This function returns the same information as \a getCandidates(),
   * but in a different format. While the latter returns a bitset of the
   * candidates, this function returns an explicit list of indices.
   *
   * \sa getCandidates()
   */
  template<typename QueryGeom>
  axom::Array<IndexType> * getCandidatesAsArray(const QueryGeom& query) const
  {
    BitsetType candidateBits = getCandidates(query);
    axom::Array<IndexType> *candidatesVec = new axom::Array<IndexType>(0, 1, candidateBits.count());
    
    for(IndexType eltIdx = candidateBits.find_first() ;
        eltIdx != BitsetType::npos;
        eltIdx = candidateBits.find_next( eltIdx) )
    {
      candidatesVec->append( eltIdx );
    }

    return candidatesVec;
  }


  /*!
   * Tests whether grid cell gridPt indexes the element with index idx
   *
   * \param [in] gridCell The cell within the ImplicitGrid that we are testing
   * \param [in] idx An element index from the IndexSet to test
   *
   * \pre \a idx should be in the IndexSet.  I.e. 0 <= idx < numIndexElements()
   * \return True if the bounding box of element \a idx overlaps
   * with GridCell \a gridCell.
   */

  bool contains(const GridCell& gridCell, IndexType idx) const
  {
    AXOM_PERF_MARK_FUNCTION("implicitgrid_contains");
    bool ret;    
    if(!m_elementSet.isValidIndex(idx) ){
      ret = false;
      return ret;
	}
    
    using reduce_pol = typename axom::execution_space< ExecSpace >::reduce_policy;
    RAJA::ReduceBitAnd< reduce_pol, unsigned int > tmp_ret(1);
    //for(int i=0 ; i< NDIMS ; ++i)
    for_all< ExecSpace >(NDIMS, [=] LAMBDA_FILLER(IndexType i)
    {
      tmp_ret &= m_bins[i].isValidIndex(gridCell[i])
               & m_binData[ i][ gridCell[i] ].test( idx);
    });
    ret = (bool)tmp_ret.get();
    return ret;
  }

private:

  /*!
   * \brief Returns the bin index in the given dimension dim
   *
   * \pre 0 <= dim < NDIMS
   */
  IndexType highestBin(int dim) const
  {
    SLIC_ASSERT(0 <= dim && dim < NDIMS);
    return m_bins[dim].size()-1;
  }

public:
  /*!
   * \brief Queries the bits that are set for dimension \a dim
   * within the range of boxes \a lower to \a upper
   *
   * \param dim The dimension to check
   * \param lower The index of the lower bin in the range (inclusive)
   * \param upper The index of the upper bin in the range (inclusive)
   *
   * \return A bitset whose bits are set if they are set in
   * any of the boxes between \a lower and \a upper for
   * dimension \a dim
   *
   * \note We perform range checking to ensure that \a lower
   * is at least 0 and \a upper is at most \a highestBin(dim)
   *
   * \sa highestBin()
   */
  BitsetType getBitsInRange(int dim, int lower, int upper) const
  {
    AXOM_PERF_MARK_FUNCTION("implicitgrid_get_bits_in_range");
    // Note: Need to clamp the gridCell ranges since the input box boundaries
    //       are not restricted to the implicit grid's bounding box
    lower = axom::utilities::clampLower(lower, IndexType() );
    upper = axom::utilities::clampUpper(upper, highestBin(dim));
    
    BitsetType bits;

    using reduce_pol = typename axom::execution_space< ExecSpace >::reduce_policy;
    RAJA::ReduceBitOr< reduce_pol, BitsetType > bit_string( m_binData[dim][lower], BitsetType(m_binData[dim][lower].size()));
    // for_all< ExecSpace >(lower+1, upper+1, AXOM_LAMBDA(IndexType i)
    //for(int i = lower+1 ; i<= upper ; ++i)
    
    for_all< ExecSpace >(lower+1, upper+1, AXOM_LAMBDA(IndexType i)
    {
      bit_string |= m_binData[dim][i];
    });

    bits = bit_string.get();
    return bits;
  }

private:

  //! The bounding box of the ImplicitGrid
  SpatialBoundingBox m_bb;

  //! A lattice to help in converting from points in space to GridCells
  LatticeType m_lattice;

  //! The amount by which to expand bounding boxes
  double m_expansionFactor;

  //! Resolution of the ImplicitGrid
  GridCell m_gridRes;

  //! The index set of the elements
  ElementSet m_elementSet;

  //! A set of bins, per dimension
  BinSet m_bins[NDIMS];

  //! The data associated with each bin
  BinBitMap m_binData[NDIMS];

  //! Tracks whether the ImplicitGrid has been initialized
  bool m_initialized;
};


} // end namespace spin
} // end namespace axom

#endif  // SPIN_IMPLICIT_GRID__HPP_
