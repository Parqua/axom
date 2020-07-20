// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Axom Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include "BitSet.hpp"

namespace axom
{
namespace slam
{

BitSet::Index const BitSet::npos = -2;

AXOM_HOST_DEVICE
void BitSet::clear()
{
  for (int i = 0 ; i < m_numWords ; ++i)
  {
    m_data[i] = 0;
  }
}

AXOM_HOST_DEVICE
void BitSet::set()
{
  if (m_numBits == 0)
  {
    return;
  }

  const Word ones = ~Word(0);
  for (int i = 0 ; i < m_numWords - 1 ; ++i)
  {
    m_data[i] = ones;
  }

  // Handle last word
  m_data[m_numWords - 1] = isLastWordFull()
                           ? ones
                           : lastWordMask();
}

AXOM_HOST_DEVICE
void BitSet::flip()
{
  if (m_numBits == 0)
  {
    return;
  }

  const Word ones = ~Word(0);
  for (int i = 0 ; i < m_numWords - 1 ; ++i)
  {
    m_data[i] ^= ones;
  }

  // Handle last word
  m_data[m_numWords - 1] ^= isLastWordFull()
                            ? ones
                            : lastWordMask();
}

AXOM_HOST_DEVICE
int BitSet::count() const
{
  int ctr = 0;

  for (int i = 0 ; i < m_numWords ; ++i)
  {
    ctr += internal::popCount(m_data[i]);
  }
  return ctr;
}

AXOM_HOST_DEVICE
bool BitSet::isValid() const
{
  bool valid = true;

  if (m_numBits < 0 || m_numWords < 0)
  {
    fprintf(stderr, "Invalid due to bits or words <\n\n");
    valid = false;
  }

  if (m_numBits == 0)
  {
    if (m_numWords != 1 || m_data[0] != Word(0))
    {
        fprintf(stderr, "Invalid due to word count: %d, or data entry: %llu  %llu \n\n", m_numWords, m_data[0], Word(0));
      valid = false;
    }
  }
  else
  {
    // check num words vs. num bits
    int expWords = (m_numBits - 1) / BITS_PER_WORD + 1;
    if (expWords != m_numWords)
      valid = false;

    // check that highest bits are not set
    if (!isLastWordFull())
    {
      const Word& lastWord = getWord(m_numBits, false);

      Word upperMask = ~lastWordMask();
      if ((lastWord & upperMask) != 0)
      {
        valid = false;
      }
    }
  }

  return valid;
}

AXOM_HOST_DEVICE
BitSet::Index BitSet::find_first() const
{
  return find_next(-1);
}

AXOM_HOST_DEVICE
BitSet::Index BitSet::find_next(Index idx) const
{
  // Handle boundary cases
  if (idx == npos || (idx+1 >= m_numBits))
  {
    return npos;
  }

  Index startWordIdx = 0;
  if (idx >= 0)
  {
    checkValidIndex(idx);

    const Index startIdx = idx + 1;
    startWordIdx = startIdx / BITS_PER_WORD;

    // Check for next set bit in current word
    const Index startOffset = startIdx % BITS_PER_WORD;

    const Word startWord = m_data[startWordIdx] >> startOffset;
    if (startWord != Word(0))
    {
      return (startWordIdx * BITS_PER_WORD)
             + internal::trailingZeros(startWord << startOffset);
    }

    ++startWordIdx;
  }

  // If not in current word, check remaining words
  for (int i = startWordIdx ; i < m_numWords ; ++i)
  {
    const Word& w = m_data[i];
    if (w != Word(0))
    {
      return (i * BITS_PER_WORD)
             + internal::trailingZeros(w);
    }
  }
  return BitSet::npos;
}

AXOM_HOST_DEVICE
BitSet& BitSet::operator|=(const BitSet& other)
{
  SLIC_ASSERT_MSG(
    size() == other.size(),
    "slam::BitSet Sizes must be the same for bit set operators."
    << " In operator|=(), BitSet has size " << size()
    << " and other BitSet has size " << other.size() << ".");

  for (int i = 0 ; i < m_numWords ; ++i)
  {
    m_data[i] |= other.m_data[i];
  }

  return *this;
}

AXOM_HOST_DEVICE
BitSet& BitSet::operator&=(const BitSet& other)
{
  SLIC_ASSERT_MSG(
    size() == other.size(),
    "slam::BitSet Sizes must be the same for bit set operators."
    << " In operator&=(), BitSet has size " << size()
    << " and other BitSet has size " << other.size() << ".");

  for (int i = 0 ; i < m_numWords ; ++i)
  {
    m_data[i] &= other.m_data[i];
  }

  return *this;
}

AXOM_HOST_DEVICE
BitSet& BitSet::operator^=(const BitSet& other)
{
  SLIC_ASSERT_MSG(
    size() == other.size(),
    "slam::BitSet Sizes must be the same for bit set operators."
    << " In operator^=(), BitSet has size " << size()
    << " and other BitSet has size " << other.size() << ".");

  for (int i = 0 ; i < m_numWords ; ++i)
  {
    m_data[i] ^= other.m_data[i];
  }

  return *this;
}

AXOM_HOST_DEVICE
BitSet& BitSet::operator-=(const BitSet& other)
{
  SLIC_ASSERT_MSG(
    size() == other.size(),
    "slam::BitSet Sizes must be the same for bit set operators."
    << " In operator-=(), BitSet has size " << size()
    << " and other BitSet has size " << other.size() << ".");

  for (int i = 0 ; i < m_numWords ; ++i)
  {
    m_data[i] &= ~other.m_data[i];
  }

  return *this;
}

AXOM_HOST_DEVICE
BitSet operator|(const BitSet & lhs, const BitSet & rhs)
{
  BitSet s(lhs);
  s |= rhs;
  return s;
}

AXOM_HOST_DEVICE
BitSet operator&(const BitSet & lhs, const BitSet & rhs)
{
  BitSet s(lhs);
  s &= rhs;
  return s;
}

AXOM_HOST_DEVICE
BitSet operator^(const BitSet & lhs, const BitSet & rhs)
{
  BitSet s(lhs);
  s ^= rhs;
  return s;
}

AXOM_HOST_DEVICE
BitSet operator-(const BitSet & lhs, const BitSet & rhs)
{
  BitSet s(lhs);
  s -= rhs;
  return s;
}

AXOM_HOST_DEVICE
bool BitSet::operator==(const BitSet & other) const
{
  if (size() != other.size() || m_numBits != other.m_numBits)
  {
    return false;
  }
  bool tester = true;
  for(int i = 0; i < m_numWords; i++){
        tester = tester && (m_data[i] == other.m_data[i]);
  }
  return tester;
}


} // end namespace slam
} // end namespace axom
