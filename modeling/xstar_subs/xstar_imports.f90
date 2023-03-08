      module xstar_imports
      contains

      subroutine ener(epi,ncn2) 
!                                                                       
!     Name: ener.f90  
!     Description:  
!     This routine sets up the energy grid
!     Grid is logarithmic with two subranges:  0.1eV-40 keV, 40keV-1MeV.
!     This structure of epi is key to operation of various other routines
!     author: T. Kallman                                                
!     List of Parameters:
!           Output:
!           epi(ncn)=energy grid (ev)
!           ncn2=length of epi
!     Dependencies:  none
!     Called by:  xstar
!                                                                       
      implicit none                           
      include './PARAM'
      real(8) epi(ncn) 
      integer numcon,numcon2,numcon3,ncn2,ll,ll2 
      real(8) ebnd1,ebnd2,ebnd2o,dele 
!                                                                       
      numcon = ncn2 
      if (numcon.gt.ncn) stop 'ncn2 too large for arry dimension' 
      if (numcon.lt.4) write (6,*) 'in ener: numcon error' 
      numcon2=max(2,ncn2/50) 
      numcon3=numcon-numcon2 
      ebnd1=0.1 
!     nb changed energy grid for H only                                 
      ebnd2=4.e+5 
!      ebnd2=4.e+1                                                      
      ebnd2o=ebnd2 
      dele=(ebnd2/ebnd1)**(1./float(numcon3-1)) 
      epi(1)=ebnd1 
!      write (lun11,*)'in ener',ncn2,numcon,numcon2,numcon3                 
      do ll=2,numcon3 
        epi(ll)=epi(ll-1)*dele 
        enddo 
      ebnd2=1.e+6 
      ebnd1=ebnd2o 
      dele=(ebnd2/ebnd1)**(1./float(numcon2-1)) 
      do ll2=1,numcon2 
        ll=ll2+numcon3 
        epi(ll)=epi(ll-1)*dele 
        enddo 
!                                                                       
      return 
      end                                           

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

      subroutine ispcg2(zremsz,epi,ncn2,enlum,lpri,lun11) 
!                                                                       
!     Name: ispcg2.f90
!     Description:
!       this subroutine calculates photon number luminosity      
!       author:  T. Kallman                   
!     Parameters:                            
!         Input:
!           zremsz:  input spectrum (erg s^-1 erg^-1 /10^38)
!           epi(ncn): photon energy grid (ev)
!           ncn2: length of epi
!           lpri: print switch
!           lun11: logical unit number for printing
!         Output:
!           enlum: photon number luminosity
!     Dependencies: none
!     called by:  rread1
!                                                                       
      use globaldata
      implicit none 
!                                                                       
      real(8) zremsz(ncn),epi(ncn) 
      integer ncn2,lpri,lun11 
      real(8) enlum 
      real(8) sum2,sum3,sum4,sum5 
      integer jk 
      integer numcon 
!                                                                       
      if (lpri.ge.1) write (lun11,*)'in ispec2' 
      numcon=ncn2 
      sum2 = 0. 
      sum3 = 0. 
      sum4 = 0. 
      sum5 = 0. 
      do jk = 1,numcon 
         if (jk.gt.1)                                                   &
     &     sum5 = sum5+(zremsz(jk)+zremsz(jk-1))                        &
     &             *(epi(jk)-epi(jk-1))/2.                              
         if ( epi(jk).ge.13.6 ) then 
            sum2 = sum2+(zremsz(jk)/epi(jk)+zremsz(jk-1)/epi(jk-1))     &
     &             *(epi(jk)-epi(jk-1))/2.                              
           if ( epi(jk).le.24.48 )                                      &
     &        sum3 = sum3+(zremsz(jk)/epi(jk)+zremsz(jk-1)/epi(jk-1))   &
     &             *(epi(jk)-epi(jk-1))/2.                              
                                                                        
         endif 
         if ((epi(jk).ge.24.48).and.(epi(jk).le.54.4))                  &
     &     sum4 = sum4+(zremsz(jk)/epi(jk)+zremsz(jk-1)/epi(jk-1))      &
     &             *(epi(jk)-epi(jk-1))/2.                              
          if (lpri.ge.1)                                                &
     &     write (lun11,*)jk,epi(jk),zremsz(jk),sum2                    
          enddo 
      enlum = sum2 
      write (lun11,*)'U(1-1.8),U(1.8-4):',sum3,sum4 
      write (lun11,*)'Lbol=',sum5*1.602197e-12 
!                                                                       
      return 
      end                                           
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

      end module xstar_imports