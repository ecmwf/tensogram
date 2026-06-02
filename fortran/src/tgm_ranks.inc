! (C) Copyright 2026- ECMWF and individual contributors.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

! Per-rank token table for the rank-generic surface (PLAN_FORTRAN.md §5.2).
! Included once per (family, dtype) with TGM_BODY set to the interface-list
! (tgm_iface.inc) or an implementation template; emits ranks 0..7. Per rank:
!   RSUF   name suffix (_r0.._r7)        DIMS   dummy dimension
!   RNK    bare rank number (0..7)       CONTIG ,contiguous (arrays)/blank
!   NEL    element count (encode side)   ALLOC  decode allocate() shape list
! Name pasting uses the traditional-cpp `/**/` form: gfortran's preprocessor
! does NOT honour the ISO `##` operator.
#define RSUF _r0
#define RNK 0
#define DIMS
#define CONTIG
#define NEL 1_c_size_t
#define ALLOC
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r1
#define RNK 1
#define DIMS (:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r2
#define RNK 2
#define DIMS (:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r3
#define RNK 3
#define DIMS (:,:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2),ext(3))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r4
#define RNK 4
#define DIMS (:,:,:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2),ext(3),ext(4))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r5
#define RNK 5
#define DIMS (:,:,:,:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2),ext(3),ext(4),ext(5))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r6
#define RNK 6
#define DIMS (:,:,:,:,:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2),ext(3),ext(4),ext(5),ext(6))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC

#define RSUF _r7
#define RNK 7
#define DIMS (:,:,:,:,:,:,:)
#define CONTIG ,contiguous
#define NEL size(a, kind=c_size_t)
#define ALLOC (ext(1),ext(2),ext(3),ext(4),ext(5),ext(6),ext(7))
#include TGM_BODY
#undef RSUF
#undef RNK
#undef DIMS
#undef CONTIG
#undef NEL
#undef ALLOC
