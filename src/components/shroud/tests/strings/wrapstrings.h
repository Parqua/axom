// wrapstrings.h
// This is generated code, do not edit
/**
 * \file wrapstrings.h
 * \brief Shroud generated wrapper for strings library
 */
// For C users and C++ implementation

#ifndef WRAPSTRINGS_H
#define WRAPSTRINGS_H

#ifdef __cplusplus
extern "C" {
#endif

// declaration of wrapped types

// splicer begin C_definition
// splicer end C_definition

const char * STR_get_string1();

void STR_get_string1_bufferify(char * SH_F_rv, int LSH_F_rv);

const char * STR_get_string2();

void STR_get_string2_bufferify(char * SH_F_rv, int LSH_F_rv);

const char * STR_get_string3();

void STR_get_string3_bufferify(char * output, int Loutput);

void STR_accept_string_const_reference(const char * arg1);

void STR_accept_string_const_reference_bufferify(const char * arg1, int Larg1);

void STR_accept_string_reference(char * arg1);

void STR_accept_string_reference_bufferify(char * arg1, int Larg1, int Narg1);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTRINGS_H
