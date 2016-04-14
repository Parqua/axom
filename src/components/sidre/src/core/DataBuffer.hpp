/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing definition of DataBuffer class.
 *
 ******************************************************************************
 */

#ifndef DATABUFFER_HPP_
#define DATABUFFER_HPP_

// Standard C++ headers
#include <vector>

// Other toolkit component headers
#include "common/CommonTypes.hpp"
#include "slic/slic.hpp"

// Sidre project headers
#include "SidreTypes.hpp"

namespace asctoolkit
{
namespace sidre
{
// using directives to make Conduit usage easier and less visible
using conduit::Node;
using conduit::Schema;

class DataStore;
class DataView;

/*!
 * \class DataBuffer
 *
 * \brief DataBuffer is a container that describes and holds data in memory.
 *
 * The DataBuffer class has the following properties:
 *
 *    - DataBuffer objects can only be created using the DataStore 
 *      createBuffer() methods. The DataBuffer ctor is private.
 *    - A DataBuffer object has a unique identifier within a DataStore,
 *      which is assigned by the DataStore when the buffer is created.
 *    - The data owned by a DataBuffer is unique to that DataBuffer
 *      object; i.e.,  DataBuffers do not share their data.
 *    - Typical usage is to describe the data a DataBuffer will hold and then
 *      allocate it by calling one of the DataBuffer allocate or
 *      reallocate methods.
 *    - A DataBuffer object maintains a collection of DataViews that
 *      refer to its data. These references are created when a DataBuffer
 *      object is attached to a DataView.
 */
class DataBuffer
{

public:

  /*!
   * Friend declarations to constrain usage via controlled access to
   * private members.
   */
  friend class DataStore;
  friend class DataGroup;
  friend class DataView;

//@{
//!  @name Basic query and accessor methods

  /*!
   * \brief Return the unique index of this buffer object.
   */
  IndexType getIndex() const
  {
    return m_index;
  }

  /*!
   * \brief Return number of views this buffer is attached to.
   */
  IndexType getNumViews() const
  {
    return m_views.size();
  }

//@}


//@{
//!  @name Methods to query and access buffer data

  /*!
   * \brief Return void-pointer to data held by buffer.
   */
  void * getVoidPtr()
  {
    return m_node.data_ptr();
  }

  /*!
   * \brief Return data held by buffer (return type is type caller assigns 
   *        return value to).
   *
   *        Note that if Buffer is not allocated, an empty Conduit 
   *        Node::Value is returned.
   */
  Node::Value getData()
  {
    if ( !isAllocated() )
    {
      SLIC_CHECK_MSG( isAllocated(), "Buffer data is not allocated.");
      return Node().value();
    }

    return m_node.value();
  }

  /*!
   * \brief Return type of data owned by this buffer object.
   */
  TypeID getTypeID() const
  {
    return static_cast<TypeID>(m_node.dtype().id());
  }

  /*!
   * \brief Return total number of data elements (of its type) owned by 
   *        this buffer object.
   */
  SidreLength getNumElements() const
  {
    return m_node.dtype().number_of_elements();
  }

  /*!
   * \brief Return total number of bytes of data owned by this buffer object.
   */
  SidreLength getTotalBytes() const
  {
    return m_node.dtype().total_bytes();
  }


  /*!
   * \brief Return true if buffer contains allocated data of > 0 bytes.
   */
  bool isAllocated() const
  {
    return (m_node.data_ptr() != ATK_NULLPTR) && (getTotalBytes() > 0);
  }

  /*!
   * \brief Return true if data description exists.  It may/may not have been
   * applied to the data yet.  ( Check isApplied() for that. )
   */
  bool isDescribed() const
  {
    return !m_node.dtype().is_empty();
  }

//@}


//@{
//!  @name Data description and allocation methods

  /*!
   * \brief Describe a buffer with data given data type and number of elements.
   *
   * To use the buffer, the data must be allocated by calling allocate().
   *
   * If buffer is already allocated or given number of elements is < 0, 
   * method is a no-op.
   *
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * describe(TypeID type, SidreLength num_elems);

  /*!
   * \brief Allocate data for a buffer.
   *
   * If the buffer is not described or already allocated the method is a no-op.
   *
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * allocate();

  /*!
   * \brief Allocate buffer with type and number of elements.
   *
   * This is equivalent to: buff->describe(type, num_elems)->allocate().
   *
   * Method is a no-op under the same conditions as either of those methods.
   *
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * allocate(TypeID type, SidreLength num_elems);

  /*!
   * \brief Reallocate data to given number of elements.
   *
   * This is equivalent to: buff->describe(type, num_elems)->allocate()
   * if the buffer is not allocated.
   *
   * If given number of elements < 0 or the buffer is not already described
   * with type information, this method is a no-op.
   *
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * reallocate(SidreLength num_elems);

  /*!
   * \brief Deallocate data in a buffer.
   *
   * If buffer is attached to views, they will be 'un-applied'.
   *
   * If the buffer is not allocated, method is a no-op.
   * 
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * deallocate();

//@}


  /*!
   * \brief Copy given number of bytes of data from source into buffer.
   *
   * If nbytes < 0 or nbytes > getTotalBytes(), or give ptr is null,
   * method is a no-op.
   *
   * \return pointer to this DataBuffer object.
   */
  DataBuffer * copyBytesIntoBuffer(const void * src, SidreLength nbytes);

  /*!
   * \brief Copy buffer description to given Conduit node.
   */
  void info(Node& n) const;

  /*!
   * \brief Print JSON description of buffer to std::cout.
   */
  void print() const;

  /*!
   * \brief Print JSON description of buffer to given output stream.
   */
  void print(std::ostream& os) const;


private:

  /*!
   *  Unimplemented ctors and copy-assignment operators.
   */
#ifdef USE_CXX11
  DataBuffer() = delete;
  DataBuffer( DataBuffer&& ) = delete;

  DataBuffer& operator=( const DataBuffer& ) = delete;
  DataBuffer& operator=( DataBuffer&& ) = delete;
#else
  DataBuffer();
  DataBuffer& operator=( const DataBuffer& );
#endif

  /*!
   *  \brief Private ctor assigns id generated by DataStore (must be 
   *         unique among buffers in DataStore.
   */
  DataBuffer( IndexType uid );

  /*!
   * \brief Private copy ctor.
   */
  DataBuffer(const DataBuffer& source );

  /*!
   * \brief Private dtor.
   */
  ~DataBuffer();

  /*!
   * \brief Private method to attach buffer to view. 
   *
   * Note: If view's buffer pointer does not match 'this', method is a no-op.
   */
  void attachToView( DataView * dataView );

  /*!
   * \brief Private method to detach buffer from view. 
   *
   * Note: If view's buffer pointer does not match 'this', method is a no-op.
   */
  void detachFromView( DataView * dataView );

  /*!
   * \brief Private method to detach buffer from all views it is attached to.
   */
  void detachFromAllViews( );

  /*!
   * \brief Private method to allocate num_bytes bytes of data and
   * return void-pointer to allocation.
   */
  void * allocateBytes(std::size_t num_bytes);

  /*!
   * \brief Private method to copy num_bytes of data from src to dst.
   */
  void copyBytes( const void * src, void * dst, size_t num_bytes );

  /*!
   * \brief Private method to delete data referenced by pointer.
   */
  void  releaseBytes(void * ptr);

  /// Buffer's unique index within DataStore object that created it.
  IndexType m_index;

  /// Container of views attached to this buffer.
  std::vector<DataView *> m_views;

  /// Conduit Node that holds buffer data.
  Node m_node;

};

} /* end namespace sidre */
} /* end namespace asctoolkit */

#endif /* DATABUFFER_HPP_ */
