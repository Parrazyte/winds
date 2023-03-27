      module globaldata
      implicit none 
!                                                                       
      include './PARAM' 
!                                                                       
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
!     compton heating data                                              
      real(8) decomp(ncomp,ncomp),ecomp(ncomp),sxcomp(ncomp) 
      end module globaldata

      module ispecg_mod
      contains

      subroutine ispecg(eptmp,zrtmp,nret,epi,ncn2,zremsz,xlum,       &
     &                  lpri,lun11)                                     
!                                                                       
!     Name: ispecg.f90
!     Description:
!       generic renormalization of initial spectrum. 
!       and mapping to epi grod
!       author:  T. Kallman                   
!     Parameters:                            
!         Input:
!           xlum: source luminosity integrated from 1-1000 Ry
!               in units of 10^38 erg/s
!           eptmp(ncn): photon energy grid (ev)
!           zrtmp:  input spectrum (erg s^-1 erg^-1 /10^38)
!           nret: length of eptmp
!           epi(ncn): photon energy grid (ev)
!           ncn2:  length of epi
!           lpri: print switch
!           lun11: logical unit number for printing
!         Output:
!           zremsz:  input spectrum (erg s^-1 erg^-1 /10^38)
!     Dependencies: none
!     called by:  rread1
!                                                               
      use globaldata
      implicit none 
!                                                                       
!                                                                       
      integer nret 
      real(8) zremsz(ncn),epi(ncn) 
      real(8) eptmp(nret),zrtmp(nret) 
      integer ncn2,lpri,lun11 
      real(8) ergsev,const,xlum 
      real(8) sum,tmp,tmpo, exp10 
      integer numcon 
      integer jlo,kl 
      real(8), dimension(:), allocatable :: zremsi
      real(8) x,epmx,epmn,zr1,zr2,ep1,ep2,alx,aly,y 
      integer jk ,ll
!
      data ergsev/1.602197e-12/ 
      save ergsev
!                                                                       
      allocate(zremsi(ncn))
!                                                                       
!        linear interpolation in log                                    
      jlo = 0 
      if (lpri.ge.1) write (lun11,*)'in ispecg:',nret                   
      if (lpri.gt.2) write (lun11,*)(ll,eptmp(ll),zrtmp(ll),ll=1,nret)
      numcon=ncn2 
      do kl = 1,numcon 
         x = epi(kl) 
         zremsi(kl) = 0. 
         epmx = max(eptmp(1),eptmp(nret)) 
         epmn = min(eptmp(1),eptmp(nret)) 
         if ( lpri.gt.2 ) write (lun11,*) kl,x,epmx,epmn 
         if ( (x.le.epmx) .and. (x.ge.epmn) ) then 
            call hunt3(eptmp,nret,x,jlo,lpri,lun11) 
            jlo = max0(jlo,1) 
            zr1 = log10(max(zrtmp(jlo+1),1.d-49)) 
            zr2 = log10(max(zrtmp(jlo),1.d-49)) 
            ep1 = log10(max(eptmp(jlo+1),1.d-49)) 
            ep2 = log10(max(eptmp(jlo),1.d-49)) 
            alx = log10(x) 
            alx = max(alx,ep2) 
            alx = min(alx,ep1) 
            aly = (zr1-zr2)*(alx-ep2)/(ep1-ep2+1.d-49) + zr2 
            y = exp10(aly) 
            zremsi(kl) = y 
            if ( lpri.gt.2 ) write (lun11,*) kl,x,jlo,zr1,zr2,          &
     &                              ep1,ep2,y                           
         endif 
         enddo 
!                                                                       
      sum = 0. 
      tmp = zremsi(1) 
      if ( lpri.gt.2 ) write (lun11,*) ' in ispecg' 
      do jk = 2,ncn2 
         tmpo = tmp 
         tmp = zremsi(jk) 
         if ( lpri.gt.2 ) write (lun11,*) jk,epi(jk),tmp,tmpo,sum 
         if ( (epi(jk).ge.13.6) .and. (epi(jk).le.1.36e+4) ) then 
            sum = sum + (tmp+tmpo)*(epi(jk)-epi(jk-1))/2. 
            endif 
         enddo 
      sum = sum*ergsev 
      const = xlum/sum 
      do jk = 1,ncn2 
         if ( lpri.gt.2 ) write (lun11,*) jk,epi(jk),zremsz(jk),        &
     &                                zremsi(jk)                        
         zremsz(jk) = zremsz(jk) + zremsi(jk)*const 
         enddo 
!
      deallocate(zremsi)
!                                                                       
      return 
      end                                           

      end module ispecg_mod