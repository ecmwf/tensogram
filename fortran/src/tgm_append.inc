! tensogram_file_append implementation template (one specific per dtype x rank).
! Included via TGM_BODY from tgm_ranks.inc. Mirrors tgm_encode.inc but targets
! the file-append entry point (no buffer returned). Tokens as in tgm_encode.inc.
   subroutine FAM/**/KS/**/RSUF(file, a, err, metadata_json, hash, encoding, filter, compression)
      type(tensogram_file),  intent(inout) :: file
      DT, target CONTIG, intent(in) :: a DIMS
      integer(c_int),        intent(out) :: err
      character(len=*), intent(in), optional :: metadata_json, hash
      character(len=*), intent(in), optional :: encoding, filter, compression
      call append_core(file%ptr, c_loc(a), byte_count(NEL, storage_size(a, kind=c_size_t)), &
                       int(shape(a), c_int64_t), DTSTR, err, &
                       metadata_json, hash, encoding, filter, compression)
   end subroutine FAM/**/KS/**/RSUF
