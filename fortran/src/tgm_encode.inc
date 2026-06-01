! tensogram_encode implementation template (one specific per dtype x rank).
! Included via TGM_BODY from tgm_ranks.inc. Tokens: FAM/KS/RSUF (name),
! DT (type/kind), DIMS/CONTIG (dummy), NEL (element count), DTSTR (wire dtype).
   subroutine FAM/**/KS/**/RSUF(a, buf, err, metadata_json, hash, encoding, filter, compression)
      DT, target CONTIG, intent(in) :: a DIMS
      type(tensogram_buffer), intent(out) :: buf
      integer(c_int),         intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call encode_core(c_loc(a), byte_count(NEL, storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), DTSTR, buf, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine FAM/**/KS/**/RSUF
