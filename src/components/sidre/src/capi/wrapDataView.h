//
// Copyright (c) 2015, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
//
// All rights reserved.
//
// This source code cannot be distributed without permission and
// further review from Lawrence Livermore National Laboratory.
//
// wrapDataView.h
// For C users and C++ implementation

#ifndef WRAPDATAVIEW_H
#define WRAPDATAVIEW_H

#include "sidre/SidreTypes.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

// declaration of wrapped types
#ifdef EXAMPLE_WRAPPER_IMPL
typedef void ATK_databuffer;
typedef void ATK_datagroup;
typedef void ATK_dataview;
#else
struct s_ATK_databuffer;
typedef struct s_ATK_databuffer ATK_databuffer;
struct s_ATK_datagroup;
typedef struct s_ATK_datagroup ATK_datagroup;
struct s_ATK_dataview;
typedef struct s_ATK_dataview ATK_dataview;
#endif

ATK_dataview * ATK_dataview_declare(ATK_dataview * self, int type, long len);

ATK_dataview * ATK_dataview_allocate(ATK_dataview * self, int type, long len);

bool ATK_dataview_has_buffer(ATK_dataview * self);

const char * ATK_dataview_get_name(const ATK_dataview * self);

ATK_databuffer * ATK_dataview_get_buffer(ATK_dataview * self);

void * ATK_dataview_get_data(ATK_dataview * self);

ATK_datagroup * ATK_dataview_get_owning_group(ATK_dataview * self);

size_t ATK_dataview_get_total_bytes(ATK_dataview * self);

#ifdef __cplusplus
}
#endif

#endif  // WRAPDATAVIEW_H
