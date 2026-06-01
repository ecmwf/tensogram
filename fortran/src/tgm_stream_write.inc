! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

! tensogram_streaming_encoder_write implementation template (per dtype x rank).
! Included via TGM_BODY from tgm_ranks.inc. Appends one data-object frame to an
! open stream. Tokens as in tgm_encode.inc (no metadata/hash here).
   subroutine FAM/**/KS/**/RSUF(enc, a, err, encoding, filter, compression)
      type(tensogram_streaming_encoder), intent(inout) :: enc
      DT, target CONTIG, intent(in) :: a DIMS
      integer(c_int),                    intent(out) :: err
      character(len=*), intent(in), optional :: encoding, filter, compression
      call stream_write_core(enc%ptr, c_loc(a), byte_count(NEL, storage_size(a, kind=c_size_t)), &
                             int(shape(a), c_int64_t), DTSTR, err, encoding, filter, compression)
   end subroutine FAM/**/KS/**/RSUF
