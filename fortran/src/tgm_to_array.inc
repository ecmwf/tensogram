! tensogram_to_array implementation template (one specific per dtype x rank).
! Included via TGM_BODY from tgm_ranks.inc. Tokens: FAM/KS/RSUF (name),
! DT (type/kind), DIMS (allocatable out), RNK (expected rank), ALLOC (allocate
! shape), DTSTR (wire dtype). The known rank replaces the F2018 SELECT RANK:
! allocate directly, then a flat copy (the reversed-shape contract makes the C
! and Fortran flat layouts coincide).
   subroutine FAM/**/KS/**/RSUF(msg, iobj, out, err)
      type(tensogram_message),         intent(in)  :: msg
      integer,                         intent(in)  :: iobj
      DT, allocatable, target, intent(out) :: out DIMS
      integer(c_int),                  intent(out) :: err
      integer(c_int64_t), allocatable :: ext(:)
      type(c_ptr) :: dptr
      DT, pointer :: src(:), dst(:)
      integer(c_size_t) :: nelem
      call object_payload(msg, iobj, DTSTR, RNK, ext, dptr, err)
      if (err /= TGM_ERROR_OK) return
      allocate(out ALLOC)
      nelem = num_elements(ext)
      call c_f_pointer(dptr,       src, [nelem])
      call c_f_pointer(c_loc(out), dst, [nelem])
      dst = src
   end subroutine FAM/**/KS/**/RSUF
