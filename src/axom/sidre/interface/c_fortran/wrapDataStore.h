// wrapDataStore.h
// This is generated code, do not edit
//
// Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
// other Axom Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
/**
 * \file wrapDataStore.h
 * \brief Shroud generated wrapper for DataStore class
 */
// For C users and C++ implementation

#ifndef WRAPDATASTORE_H
#define WRAPDATASTORE_H

#include "typesSidre.h"
#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include "axom/sidre/core/SidreTypes.hpp"
#else
#include <stdbool.h>
#include <stddef.h>
#include "axom/sidre/interface/SidreTypes.h"
#endif

// splicer begin class.DataStore.CXX_declarations
// splicer end class.DataStore.CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin class.DataStore.C_declarations
// splicer end class.DataStore.C_declarations

SIDRE_DataStore* SIDRE_DataStore_new(SIDRE_DataStore* SHC_rv);

void SIDRE_DataStore_delete(SIDRE_DataStore* self);

SIDRE_Group* SIDRE_DataStore_get_root(SIDRE_DataStore* self,
                                      SIDRE_Group* SHC_rv);

size_t SIDRE_DataStore_get_num_buffers(const SIDRE_DataStore* self);

SIDRE_Buffer* SIDRE_DataStore_get_buffer(SIDRE_DataStore* self,
                                         SIDRE_IndexType idx,
                                         SIDRE_Buffer* SHC_rv);

SIDRE_Buffer* SIDRE_DataStore_create_buffer_empty(SIDRE_DataStore* self,
                                                  SIDRE_Buffer* SHC_rv);

SIDRE_Buffer* SIDRE_DataStore_create_buffer_from_type(SIDRE_DataStore* self,
                                                      int type,
                                                      SIDRE_IndexType num_elems,
                                                      SIDRE_Buffer* SHC_rv);

void SIDRE_DataStore_destroy_buffer(SIDRE_DataStore* self, SIDRE_IndexType id);

bool SIDRE_DataStore_generate_blueprint_index(SIDRE_DataStore* self,
                                              const char* domain_path,
                                              const char* mesh_name,
                                              const char* index_path,
                                              int num_domains);

bool SIDRE_DataStore_generate_blueprint_index_bufferify(SIDRE_DataStore* self,
                                                        const char* domain_path,
                                                        int Ldomain_path,
                                                        const char* mesh_name,
                                                        int Lmesh_name,
                                                        const char* index_path,
                                                        int Lindex_path,
                                                        int num_domains);

void SIDRE_DataStore_print(const SIDRE_DataStore* self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPDATASTORE_H
