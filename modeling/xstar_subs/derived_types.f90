      module derived_types

      include './PARAM' 
      
      public:: master_data,derived_pointers,level_temp
      
!     global xstar data
!     master data

      TYPE :: master_data
        integer, allocatable, dimension(:) :: idat1 ! integer data
        real(8), allocatable, dimension(:) :: rdat1  ! real data
        integer, allocatable, dimension(:,:) :: nptrs ! pointer data
        character(1), allocatable, dimension(:) :: kdat1 ! character data
      END TYPE master_data
      TYPE(master_data) :: masterdata
      TYPE :: derived_pointers
        integer, allocatable, dimension(:) :: npar   !    pointers to master data
        integer, allocatable, dimension(:) :: npnxt   !    pointers to master data
        integer, allocatable, dimension(:) :: npfirst !    pointers to master data
        integer, allocatable, dimension(:,:) :: npfi    !    pointers to master data first record for ion
        integer, allocatable, dimension(:,:) :: npfe  !   pointers to master data first record from element
        integer, allocatable, dimension(:) :: nplin   ! pointers to line data
        integer, allocatable, dimension(:) :: nplini  ! pointers to line data
        integer, allocatable, dimension(:) :: idst1   ! pointer to lower level
        integer, allocatable, dimension(:) :: idst2   ! pointer to upper level
        integer, allocatable, dimension(:) :: npcon
        integer, allocatable, dimension(:) :: npconi2 
        integer, allocatable, dimension(:) :: npconi
        integer, allocatable, dimension(:,:) :: npilev
        integer, allocatable, dimension(:) :: npilevi
        integer, allocatable, dimension(:) :: nlevs
      END TYPE derived_pointers
      TYPE(derived_pointers) :: derivedpointers
      TYPE :: level_temp
        sequence
        real(8) :: rlev(10,ndl) 
        integer:: ilev(10,ndl),nlpt(ndl),iltp(ndl) 
        character(1) :: klev(100,ndl) 
      END TYPE level_temp
      TYPE(level_temp) :: leveltemp
      end module derived_types