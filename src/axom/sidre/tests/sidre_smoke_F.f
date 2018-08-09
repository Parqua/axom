!
! Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
! 
! Produced at the Lawrence Livermore National Laboratory
! 
! LLNL-CODE-741217
! 
! All rights reserved.
!
! This file is part of Axom.
! 
! For details about use and distribution, please read axom/LICENSE.
!

module sidre_smoke_test
  use fruit
  use axom_sidre
  implicit none

contains
!------------------------------------------------------------------------------

  subroutine create_datastore
    type(SidreDataStore) ds

    call set_case_name("create_datastore")

    ds = datastore_new()
    call ds%delete()

    call assert_true(.true.)
  end subroutine create_datastore

!------------------------------------------------------------------------------

  subroutine valid_invalid
    type(SidreDataStore) ds
    type(SidreGroup) root
    integer idx
    character(10) name

    call set_case_name("valid_invalid")

    ds = datastore_new()

    idx = 3;
    call assert_true(idx /= invalid_index, "invalid_index does not compare")

    name = "foo"
    call assert_true(name_is_valid(name), "name_is_valid")

    root = ds%get_root()

    call assert_true(root%get_group_name(idx) == " ", &
         "error return from get_group_name")
    call assert_true(root%get_group_index(name) == invalid_index, &
         "root%get_group_index(name) == invalid_index")

    call ds%delete()
  end subroutine valid_invalid

!------------------------------------------------------------------------------
end module sidre_smoke_test
!------------------------------------------------------------------------------

program fortran_test
  use iso_c_binding
  use fruit
  use sidre_smoke_test
  implicit none

  logical ok

  call init_fruit

  call create_datastore
  call valid_invalid

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif
end program fortran_test
