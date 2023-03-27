      subroutine ispecgg(xlum,epi,ncn2,zremsz,                       &
     &               lpri,lun11)                                        
!                                                                       
!     Name: ispecgg.f90
!     Description:
!       this subroutine renormalizes the initial spectrum. 
!       brems stores the flux to be used    
!       author:  T. Kallman                   
!     Parameters:                            
!         Input:
!           xlum: source luminosity integrated from 1-1000 Ry
!               in units of 10^38 erg/s
!           epi(ncn): photon energy grid (ev)
!           ncn2: length of epi
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
      real(8) epi(ncn),zremsz(ncn) 
      integer numcon,ncn2,i 
      real(8) ergsev,sum,const,xlum 
      integer lpri, lun11 
!                                                                       
      data ergsev/1.602197e-12/ 
      save ergsev
!                                                                       
      numcon=ncn2 
      sum=0. 
      if (lpri.gt.1) write (lun11,*)'in ispec',xlum 
      do i=1,numcon 
         if (lpri.gt.1) write (lun11,*)i,epi(i),zremsz(i) 
         if ((epi(i).ge.13.6).and.(epi(i).le.1.36e+4)                   &
     &        .and.(i.gt.1))                                            &
     &    sum=sum+(zremsz(i)+zremsz(i-1))*(epi(i)-epi(i-1))/2.          
         enddo 
!                                                                       
      const=xlum/sum/ergsev 
      do i=1,numcon 
         zremsz(i)=zremsz(i)*const 
         if (lpri.gt.1)                                                 &
     &        write (lun11,*)i,epi(i),const,zremsz(i)                   
         enddo 
!                                                                       
      return 
      END                                           
