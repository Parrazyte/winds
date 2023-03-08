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
      module times
      integer ntp
      parameter (ntp=99)
      real tread,tloop,tfunc,trates1,thcor,trates2,tucalc(ntp),theat
      integer ncall(ntp)
      end module times


      module xstar_mod
      contains

      subroutine xstar(plop) 
!                                                                       
!      based on attenuate
!
      use globaldata
      use times
      implicit none 
      integer nzones
      parameter (nzones=100)
!                                                                       
      TYPE :: level_temp
        sequence
        real(8) :: rlev(10,ndl) 
        integer:: ilev(10,ndl),nlpt(ndl),iltp(ndl) 
        character(1) :: klev(100,ndl) 
      END TYPE level_temp
      TYPE(level_temp) :: leveltemp
!     global xstar data
!     line luminosities                                                 
      real(8), dimension(:,:), allocatable :: elum,elumo
!     line emissivities                                                 
      real(8), dimension(:,:), allocatable :: rcem
!     line opacities                                                    
      real(8), dimension(:), allocatable ::  oplin
      real(8), dimension(:,:), allocatable :: fline
      real(8), dimension(:), allocatable :: flinel 
!     line optical depths                                               
      real(8), dimension(:,:), allocatable ::  tau0
!     energy bins                                                       
      real(8) epi(ncn) 
      real(8) epim(ncn) 
!      continuum lum                                                    
      real(8), dimension(:,:), allocatable ::  zrems, zremso
      real(8), dimension(:), allocatable :: zremsz
!     continuum optical depths                                          
      real(8), dimension(:,:), allocatable ::  dpthc,dpthcont
!     continuum flux                                                    
      real(8), dimension(:), allocatable ::  bremsa,bremsint 
      real(8), dimension(:), allocatable ::  bremsam
!     continuum emissivities                                            
      real(8), dimension(:,:), allocatable ::  rccemis
      real(8), dimension(:), allocatable ::  brcems
!     continuum opacities                                               
      real(8), dimension(:), allocatable :: opakc,opakcont
!     level populations                                                 
      real(8), dimension(:), allocatable :: xilevg,bilevg,rnisg
      real(8), dimension(:), allocatable :: cabab,opakab
      real(8), dimension(:,:), allocatable :: cemab,elumab,elumabo
      real(8), dimension(:,:), allocatable :: tauc
!     ion abundances                                                    
      real(8) xii(nni) 
!     heating and cooling                                               
      real(8) htt(nni),cll(nni) 
      real(8) htt2(nni),cll2(nni) 
      real(8) rrrt(nni),pirt(nni) 
!     element abundances                                                
      real(8) abel(nl),ababs(nl) 
!     the atomic data creation date                                     
      character(63) atcredate 
!     pprint arrays
      real(8), dimension(:,:), allocatable ::  zrtmp
      real(8), dimension(:,:), allocatable ::  zrtmpcol
      real(8), dimension(:,:), allocatable ::  zrtmpc
      real(8), dimension(:,:), allocatable ::  zrtmph
      real plop
!              
!     local variables                                                   
!     state variables                                                   
      real(8) p,rdel,r19,xi,xcol,zeta,rdelo,r,t,xpx,delr,xpx0,r0 
!     heating-cooling variables                                         
      real(8) httot,cltot,htcomp,clcomp,clbrems,elcter,cllines,          &
     &     clcont,hmctot,httot2,cltot2,htfreef
!     limits on ion indeces vs element
      integer mml(nl),mmu(nl)
!     switch for absolute value of line wavelengths
      integer llinabs
!
!     input parameters                                                  
      character(16) knam,knam2,knam3,knam4 
      character(80) kmodelname,specfile,spectype 
      real(8) enlum,emult,taumax,xeemin,xlum,rmax,xpxcol,trad 
      real(8) cfrac,critf,vturbi,xlum2,xee,radexp 
      integer lcdd,ncn2,ncn2m
!     variables associated with thermal equilibrium solution            
      integer ntotit,lnerrd 
!     switches                                                          
      integer lprisv,lpri2,lpri3
      integer  nnmax,nlimd,lunlog,nloopctl,numrec,npass,lfix 
!     temporary for xwrite                                              
      character(133) tmpst 
!     temporary for spectrum                                            
      real(8) eptmp(ncn),zrtmpp(ncn) 
      integer nlprnt(19),nlnprnt 
!     temporary integers                                                
      integer mm,kk,ldir,jk,jkp
      integer nlsvn,ncsvn 
      integer istatus, iunit,iunit2,numcon2,iunit3,iunit4 
      integer iunito,iunit2o,iunit3o,iunit4o,ierr
!     times                                                             
      real(8) tinf,ttot,t1s,t2s 
      integer np2,lun40 
      integer lprid,lpril,lpri,lwri,lpriu,nlimdt,lpris 
      integer lpril2 
!     storing info for parameters                                       
      character(20) parname(55) 
      character(10) partype(55) 
      real(8) parms(55) 
      character(30) parcomm(55) 
      integer nparms, specunit
!     temporary line pointers                                           
      integer, dimension(:), allocatable :: nlin
      real(8), dimension(:), allocatable :: elin
      real(8), dimension(:), allocatable :: errc
      real(8) eliml,elimh 
      real(8) ectt 
      real(8) rnew,dennew 
      integer nry,nbinc
      character(4) karg(20)
!
      integer status
!                                                                       
!     local definitions                                                 
!                                                                       
!                                                                       
!     Warning!!  #11 & #7 must be run before #5 since #5 changes        
!                the variable zrtmp                                     
!                Also make sure that if you change the number of        
!                entries in nlprnt, that you also update nlnprnt        
!                and the real(4) statement for nlprnt                    
      data nlprnt/22,11,1,23,24,16,27,15,19,5,14,21,7,10,               &
     &            26,0,4,6,18/                                        
!     &            21,7,18,27,8,26/                                      
!                                                                       
!     Parameter Names                                                   
      data parname/'cfrac','temperature',                               &
     &   'lcpres','pressure','density','spectrum',                      &
     &   'spectrum_file','spectun','trad',                              &
     &   'rlrad38','column','rlogxi',                                   &
     &   'nsteps','niter','lwrite',                                     &
     &   'lprint','lstep',                                              &
     &   'habund','heabund',                                            &
     &   'liabund','beabund','babund','cabund',                         &
     &   'nabund','oabund','fabund','neabund',                          &
     &   'naabund','mgabund','alabund',                                 &
     &   'siabund','pabund','sabund',                                   &
     &   'clabund','arabund','kabund',                                  &
     &   'caabund','scabund','tiabund',                                 &
     &   'vabund','crabund','mnabund ',                                 &
     &   'feabund','coabund','niabund',                                 &
     &   'cuabund','znabund','emult','taumax','xeemin',                 &
     &   'critf','vturbi','npass','modelname',                          &
     &   'loopcontrol'/                                                 
      data partype/'real','real',                                       &
     &    'integer','real','real','string',                             &
     &    'string','integer','real',                                    &
     &    'real','real','real',                                         &
     &    'integer','integer','integer',                                &
     &    'integer','integer',                                          &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real',                                                &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real',                                         &
     &    'real','real','real','integer','string',                      &
     &    'integer'/                                                    
      data parcomm/' ','Units of 10**4 K',                              &
     &     '1=yes, 0=no','dynes/cm**2','cm**(-3)',' ',                  &
     &     ' ','0=energy, 1=photons','or alpha',                        &
     &     '/10**38 erg/sec','cm**(-2)',' ',                            &
     &     ' ',' ','1=yes, 0=no',                                       &
     &     '1=yes, 0=no',' ',                                           &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',                                                     &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',' ',                                                 &
     &     ' ',' ',                                                     &
     &     ' '/                                                         
!
      allocate(masterdata%rdat1(nrdat1))
      allocate(masterdata%idat1(nidat1))
      allocate(masterdata%kdat1(nkdat1))
      allocate(masterdata%nptrs(nptt,ndat2))
      allocate(derivedpointers%npar(ndat2))
      allocate(derivedpointers%npnxt(ndat2))
      allocate(derivedpointers%npfirst(ntyp))
      allocate(derivedpointers%npfi(ntyp,nni))
      allocate(derivedpointers%npfe(nl,ntyp))
      allocate(derivedpointers%nplin(nnnl))
      allocate(derivedpointers%nplini(ndat2))
      allocate(derivedpointers%npcon(nnml))
      allocate(derivedpointers%npconi(ndat2))
      allocate(derivedpointers%npconi2(ndat2))
      allocate(derivedpointers%npilev(nd,nni))
      allocate(derivedpointers%npilevi(nnml))
      allocate(derivedpointers%nlevs(nni))
      allocate(elum(2,nnnl))
      allocate(elumo(2,nnnl))
      allocate(rcem(2,nnnl))
      allocate(oplin(nnnl))
      allocate(fline(2,nnnl))
      allocate(flinel(ncn))
      allocate(tau0(2,nnnl))
      allocate(zrems(5,ncn))
      allocate(zremso(5,ncn))
      allocate(zremsz(ncn))
      allocate(dpthc(2,ncn))
      allocate(dpthcont(2,ncn))
      allocate(bremsa(ncn))
      allocate(bremsam(ncn))
      allocate(bremsint(ncn))
      allocate(rccemis(2,ncn))
      allocate(brcems(ncn))
      allocate(opakc(ncn))
      allocate(opakcont(ncn))
      allocate(bilevg(nnml))
      allocate(rnisg(nnml))
      allocate(xilevg(nnml))
      allocate(cemab(2,nnml))
      allocate(cabab(nnml))
      allocate(opakab(nnml))
      allocate(elumab(2,nnml))
      allocate(elumabo(2,nnml))
      allocate(tauc(2,nnml))
      allocate(elin(nnnl))
      allocate(errc(nnml))
      allocate(nlin(nnnl))
      allocate(zrtmp(999,3999))
      allocate(zrtmpcol(999,1))
      allocate(zrtmpc(999,3999))
      allocate(zrtmph(999,3999))
!
!                                                                       
      nparms=55 
!                                                                       
!     Allocate temporary and logging files                              
      call getlunx(lunlog) 
      open(unit=lunlog,file='xout_step.log',status='unknown') 
!                                                                       
      call remtms(t1s) 
!                                                                       
!     opening message                                                   
      write (tmpst,*)'xstar version 2.59' 
      call xwrite(tmpst,10) 
!                                                                       
!     default parameter values                                          
!                                                                       
      trad=1. 
      xlum=1. 
      lpri=0 
      lwri=0 
      r=0. 
      r0=r 
      t=1. 
      xpx=1000. 
      xpx0=xpx 
      p=0.03 
      lcdd=1 
      numrec=2 
      npass=0 
      nnmax=1 
      nlimd=0 
      rmax=1. 
      xpxcol=1.e+16 
      zeta=0. 
      xi=10.**zeta 
      lfix=0 
      ncn2=max(ncn,999) 
      call ener(epi,ncn2) 
      ncn2m=999
      call ener(epim,ncn2m) 
      do mm=1,ncn2 
        zremsz(mm)=0. 
        enddo 
      cfrac=0. 
      emult=0.75 
      taumax=5. 
      xeemin=0.1 
      nloopctl=0 
      critf=1.d-8 
      vturbi=1. 
      xcol=0. 
      radexp=0. 
      nry=nbinc(13.6d0,epi,ncn2)+2
!                                                                       
!     read in parameter values                                          
      call rread1(trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,numrec,npass,   &
     & nlimd,rmax,xpxcol,xi,zeta,lfix,                                  &
     & lunlog,abel,cfrac,emult,taumax,xeemin,spectype,specfile,         &
     & specunit,kmodelname,nloopctl,critf,vturbi,eptmp,zrtmpp,numcon2,  &
     &     ncn2,radexp)                                                 
!            
      if (radexp.lt.-99.) then 
        call getlunx(lun40) 
        open(lun40,iostat=ierr,file='density.dat') 
        if (ierr.ne.0) stop 'missing density file' 
        read (lun40,*,iostat=ierr)rnew,dennew 
        r=rnew 
        xpx=dennew 
        endif 
!                                                                       
      lprisv=lpri 
      lpri=0
      call ener(epi,ncn2) 
      do mm=1,ncn2 
        zremsz(mm)=0. 
        enddo 
!     
!     take absolute value of line wavelengths
      llinabs=1
      call xstarsetup(lnerrd,nlimd,                                     &
     &       lpri,lprid,lunlog,llinabs,tinf,critf,                      &
     &       t,trad,r,delr,xee,xpx,ababs,abel,cfrac,xlum,p,lcdd,        &
     &       epi,ncn2,bremsa,bremsint,atcredate,                        &
     &       epim,ncn2m,bremsam,                                        &
     &       zrems,zremso,zremsz,                                       &
     &       tau0,dpthc,dpthcont,tauc,                                  &
     &       np2,ncsvn,nlsvn,                                           &
     &       ntotit,                                                    &
     &       xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,elcter, &
     &       httot2,cltot2,                                             &
     &       xilevg,bilevg,rnisg,elum,elumo,fline,flinel,               &
     &       rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,            &
     &       cabab,opakab,elumab,elumabo,nlin,elin,errc)                                    
!
      xee=1.21 
      xi=10.**zeta 
!                                                                       
!     Write the parameter list to the FITS file                         
!     When changing this list, make sure nparms, parname, partype,      
!     and parcomm are also properly updated.  Watch out for the         
!     model name which is currently parcomm(37)                         
      call pprint(3,1,trad,xlum,lwri,lprisv,r,t,xpx,p,lcdd,             &
     &        numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,        &
     &        zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,           &
     &        spectype,specfile,specunit,kmodelname,nloopctl,           &
     &        nparms,parname,partype,parms,parcomm,atcredate,           &
     &        lunlog,tinf,xcol,vturbi,critf,radexp,                     &
     &        delr,rdel,enlum,xee,ababs,                                &
     &        bremsa,bremsint,tau0,dpthc,tauc,                          &
     &        ncsvn,nlsvn,                                              &
     &        ntotit,lnerrd,                                            &
     &        xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,       &
     &        cllines,clcont,htcomp,clcomp,clbrems,htfreef,             &
     &        httot2,cltot2,                                            &
     &        xilevg,bilevg,rnisg,                                      &
     &        rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,    &
     &        cabab,elumab,elum,zrems,fline,flinel,                     &
     &        zrtmp,zrtmpcol,zrtmph,zrtmpc)                                  
!                                                                       

!     set up and initialize                                             
      rdel = 0. 
      call pprint(2,1,trad,xlum,lwri,lprisv,r,t,xpx,p,lcdd,             &
     &        numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,        &
     &        zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,           &
     &        spectype,specfile,specunit,kmodelname,nloopctl,           &
     &        nparms,parname,partype,parms,parcomm,atcredate,           &
     &        lunlog,tinf,xcol,vturbi,critf,radexp,                     &
     &        delr,rdel,enlum,xee,ababs,                                &
     &        bremsa,bremsint,tau0,dpthc,tauc,                          &
     &        ncsvn,nlsvn,                                              &
     &        ntotit,lnerrd,                                            &
     &        xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,       &
     &        cllines,clcont,htcomp,clcomp,clbrems,htfreef,             &
     &        httot2,cltot2,                                            &
     &        xilevg,bilevg,rnisg,                                      &
     &        rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,    &
     &        cabab,elumab,elum,zrems,fline,flinel,                     &
     &        zrtmp,zrtmpcol,zrtmph,zrtmpc)                                  
!                                                                     
      write (lunlog,*) 'running ...' 
      write (tmpst,*) 'running ...' 
!
!     Entering the main processing loop                                 
      if (numrec.le.0) npass=1 
      do kk=1,npass 
!                                                                       
        if ((lwri.gt.0).or.(npass.gt.1)) then 
!         Open and initialize FITS file for detailed ionic data         
          knam='xout_detail.fits' 
          call fnappend(knam,kk) 
          iunito=iunit 
          call fheader(iunit,knam,atcredate,kmodelname,istatus) 
          knam2='xout_detal2.fits' 
          call fnappend(knam2,kk) 
          iunit2o=iunit2 
          call fheader(iunit2,knam2,atcredate,kmodelname,istatus) 
          if (istatus .gt. 0)call printerror(lunlog,istatus) 
          knam3='xout_detal3.fits' 
          call fnappend(knam3,kk) 
          iunit3o=iunit3 
          call fheader(iunit3,knam3,atcredate,kmodelname,istatus) 
          knam4='xout_detal4.fits' 
          call fnappend(knam4,kk) 
          iunit4o=iunit4 
          call fheader(iunit4,knam4,atcredate,kmodelname,istatus) 
          call fparmlist(iunit,1,kmodelname,nparms,parname,partype,  &
     &               parms,parcomm,nloopctl,istatus,lunlog)             
          call fparmlist(iunit2,1,kmodelname,nparms,parname,partype, &
     &               parms,parcomm,nloopctl,istatus,lunlog)             
          call fparmlist(iunit3,1,kmodelname,nparms,parname,partype, &
     &               parms,parcomm,nloopctl,istatus,lunlog)             
          call fparmlist(iunit4,1,kmodelname,nparms,parname,partype, &
     &               parms,parcomm,nloopctl,istatus,lunlog)             
          endif 
!                                                                       
        if (spectype.eq.'file    ')                                     &
     &    call ispecg(eptmp,zrtmpp,numcon2,epi,ncn2,zremsz,xlum,        &
     &                lpri,lunlog)                                    
        xlum2=1. 
        if (spectype.eq.'bbody   ') then 
          call starf(trad,xlum2,epi,ncn2,zremsz,lpri,lunlog) 
          endif 
        if (spectype.eq.'pow     ')                                     &
     &    call ispec4(trad,xlum,epi,ncn2,zremsz,lpri,lunlog)          
        if (spectype.eq.'brems   ')                                     &
     &    call ispec(trad,xlum,epi,ncn2,zremsz,lpri,lunlog)           
        call ispecgg(xlum,epi,ncn2,zremsz,                              &
     &             lpri,lunlog)                                       
        call ispcg2(zremsz,epi,ncn2,enlum,lpri,lunlog) 
!                                                                     
      call init(lunlog,abel,bremsa,bremsam,bremsint,tau0,dpthc,dpthcont,&
     &   tauc,                                                          &
     &   xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,httot2,cltot2,     &
     &   cllines,clcont,htcomp,clcomp,clbrems,                          &
     &   xilevg,rcem,oplin,rccemis,brcems,opakc,opakcont,               &
     &   cemab,cabab,opakab,elumab,elumabo,elum,elumo,                  &
     &   zrems,zremso,fline,flinel)                                     
      lpri=0 
!                                                                     
        ldir=(-1)**kk 
        write (lunlog,*) ' ' 
        write (tmpst,*) ' ' 
        call xwrite(tmpst,10) 
        write (lunlog,*) 'pass number=',kk,ldir 
        write (tmpst,*) 'pass number=',kk,ldir 
        call xwrite(tmpst,10) 
!                                                                       
!     Print a heading for this pass                                   
      call pprint(17,1,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,           &
     &      numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,          &
     &      zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,             &
     &      spectype,specfile,specunit,kmodelname,nloopctl,             &
     &      nparms,parname,partype,parms,parcomm,atcredate,             &
     &      lunlog,tinf,xcol,vturbi,critf,radexp,                       &
     &      delr,rdel,enlum,xee,ababs,                                  &
     &      bremsa,bremsint,tau0,dpthc,tauc,                            &
     &      ncsvn,nlsvn,                                                &
     &      ntotit,lnerrd,                                              &
     &      xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,         &
     &      cllines,clcont,htcomp,clcomp,clbrems,htfreef,               &
     &       httot2,cltot2,                                             &
     &        xilevg,bilevg,rnisg,                                      &
     &      rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,      &
     &      cabab,elumab,elum,zrems,fline,flinel,                       &
     &      zrtmp,zrtmpcol,zrtmph,zrtmpc)                                  
!                                                                     
      do jk=1,ncn2 
        zrems(1,jk)=zremsz(jk) 
        zremso(1,jk)=zremsz(jk) 
        enddo 
!
!       main loop, step thru radius zones                               
        jkp=0 
        rdelo=0. 
        rdel=0. 
        ierr=0 
        do while ((((kk.eq.1).and.(xcol.lt.xpxcol).and.(xee.gt.xeemin)  &
     &         .and.(t.gt.tinf*(0.99)).and.(numrec.gt.0))               &
     &         .or.((kk.gt.1).and.(jkp.lt.numrec))).and.(ierr.eq.0))    
                                                                        
!                                                                       
!         save for variable density                                     
          if ((kk.eq.1).and.(lcdd.eq.1).and.(jkp.eq.0)) then 
            xpx0=xpx 
            r0=r 
            endif 
                                                                        
          jkp=jkp+1 
          if (jkp.gt.3999) stop 'too many steps: buffer filled' 
!                                                                       
!         step forward                                                  
          if ((kk.gt.1).and.(kk.le.npass)) then 
             jk=numrec+1-jkp 
             lpriu=0 
             rdelo=rdel 
             if (ldir.gt.0) then 
                 nlimdt=0 
               else 
                 nlimdt=nlimd 
               endif 
             lpriu=0 
             call unsavd(jk+2,ldir,                                  &
     &       lpri,iunito,iunit2o,iunit3o,iunit4o,                       &
     &       t,p,r,rdel,delr,xcol,xee,xpx,zeta,                         &
     &       xilevg,rnisg,                                              &
     &       rcem,oplin,tau0,                                           &
     &       cemab,cabab,opakab,tauc,                                   &
     &       epi,ncn2,zrems,dpthc,opakc,rccemis,                        &
     &       lunlog,status)                                             
             endif 
          if (kk.eq.1) then 
             nlimdt=nlimd 
             delr=0. 
             ectt=1. 
             jk=jkp 
             lpris=0
             if (jk.gt.1)                                               &
     &         call step(ectt,emult,epi,ncn2,opakc,rccemis,fline,       &
     &           zrems,lpris,delr,dpthc,r,                              &
     &           xpxcol,xcol,xpx,taumax,numrec,lunlog)                  
            endif 
!
!          calculate flux                                                
           r19=r*(1.e-19) 
           xi=xlum/r19/r19/xpx 
           zeta=log10(xi) 
           lpri3=0
           call trnfrc(lpri3,lunlog,ldir,                               &
     &      r,xpxcol,xpx,                                               &
     &      epi,ncn2,zremsz,dpthc,opakc,                                &
     &      zrems,bremsa,bremsint)                            
!                 
                                                    
          if (t.lt.tinf*(1.02)) then 
            t=tinf*(1.01) 
            endif 
!
!         calculate temperature, ionization, etc.                       
          lpri2=0
!          if (jkp.eq.2) lpri2=1
!          if (kk.gt.1) lpri2=1
          lprid=0
          call xstarcalc(lpri2,lnerrd,nlimd,                            &
     &            lpri3,lprid,lunlog,tinf,vturbi,critf,                 &
     &            t,trad,r,delr,xee,xpx,ababs,cfrac,p,lcdd,zeta,        &
     &            mml,mmu,                                              &
     &            epi,ncn2,bremsa,bremsint,                             &
     &            epim,ncn2m,bremsam,                                   &
     &            leveltemp,                                            &
     &            tau0,tauc,                                            &
     &            np2,ncsvn,nlsvn,                                      &
     &            ntotit,                                               &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            elcter,                                               &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &             xilevg,bilevg,rnisg,                                 &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,       &
     &            cabab,opakab,fline,flinel,elin,errc)                
!
!           ekt=0.861707*t
!           do mm=1,ncn2
!             if ((dpthc(1,mm).ge.0.67).and.(dpthco(mm).lt.0.67))        &
!     &          then
!              constant is h^3c^2 in eV
!               bbnu=2.*epi(mm)**3/max(1.e-34,expo(epi(mm)/ekt)-1)       &
!     &              /(6.366e-23)
!               zrems6(mm)=bbnu
!               endif
!             enddo
!
!          do tranfer.  assumes comp2 and bremem have been called             
!          already                                                           
           call heatt(lpri2,lunlog,                                     &
     &       t,r,cfrac,delr,xee,xpx,abel,                               &
     &       epi,ncn2,bremsa,                                           &
     &       leveltemp,                                                 &
     &       ncsvn,nlsvn,                                               &
     &       zrems,zremso,elumab,elumabo,elum,elumo,                    &
     &       rcem,rccemis,opakc,opakcont,cemab,flinel,                  &
     &       brcems)
!                                                                       
          call pprint(9,jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,         &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
!
!
!         If this is the final pass over all zones                      
          if (kk.eq.npass) then 
!           Add the abundances information to the output array zrtmp    
              call pprint(12,jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd, &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
          endif
!         Write radial profile FITS extension in here...                
          if ((lwri.gt.0).or.(npass.gt.1))                              &
     &         call  savd(jkp,ldir,                                     &
     &            lpri,iunit,iunit2,iunit3,iunit4,                      &
     &            np2,nlsvn,                                            &
     &            t,p,r,rdel,delr,xcol,xee,xpx,zeta,abel,               &
     &            xilevg,rnisg,                                         &
     &            rcem,oplin,tau0,                                      &
     &            cemab,cabab,opakab,tauc,                              &
     &            epi,ncn2,zrems,dpthc,opakc,rccemis,                   &
     &            lunlog,status)                                        
!         new position etc                                              
          if (radexp.lt.-99.) then 
              read (lun40,*,iostat=ierr)rnew,dennew 
              delr=rnew-r 
              if (delr.lt.0.) stop 'radius error' 
              r=rnew 
              xpx=dennew 
!              write (lunlog,*)'rnew,dennew:',rnew,dennew,ierr          
            else 
              r=r+delr 
              if (lcdd.eq.1)                                            &
     &          xpx=xpx0*(r/r0)**radexp                                 
            endif 
          rdel=rdel+delr 
          xcol=xcol+xpx*delr 
!                                                                       
!         transfer                                                      
          eliml=1. 
          elimh=1.0e6 
          lpril2=0
!          do mm=1,ncn2
!            dpthco(mm)=dpthc(1,mm)
!            enddo
          call stpcut(ldir,lpril2,lunlog,                               &
     &           ncsvn,nlsvn,                                           &
     &           epi,ncn2,opakc,opakcont,oplin,opakab,delr,             &
     &           dpthc,dpthcont,tau0,tauc) 
          call trnfrn(lpri,lunlog,                                      &
     &           nlsvn,ncsvn,ncn2,                                      &
     &           zrems,zremso,elumab,elumabo,elum,elumo)                    
!
!
!          All done looping over the radial zones                        
           enddo 
!
          if (kk.eq.1) numrec=jkp 
!
!       another printout to get the last step                           
          call pprint(9,jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,      &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
!       If this is the final pass over all zones                        
        if (kk.eq.npass) then 
!         Add the abundances information to the output array zrtmp      
          call pprint(12,jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,     &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
            endif 
          if ((lwri.gt.0).or.(npass.gt.1))                              &
     &     call savd(jkp+1,ldir,                                     &
     &       lpri,iunit,iunit2,iunit3,iunit4,                           &
     &       np2,nlsvn,                                                 &
     &       t,p,r,rdel,delr,xcol,xee,xpx,zeta,abel,                    &
     &       xilevg,rnisg,                                              &
     &       rcem,oplin,tau0,                                           &
     &       cemab,cabab,opakab,tauc,                                   &
     &       epi,ncn2,zrems,dpthc,opakc,rccemis,                        &
     &       lunlog,status)                                             
        if (kk.gt.1) then 
          call fitsclose(lunlog,iunito,istatus) 
          call fitsclose(lunlog,iunit2o,istatus) 
          call fitsclose(lunlog,iunit3o,istatus) 
          call fitsclose(lunlog,iunit4o,istatus) 
          endif 
                                                                        
        lpriu=0 
!                                                                       
!       End of the main iteration loop                                  
        enddo 

9005  continue
!                                                                       
!                                                                       
      write (lunlog,*)' ' 
      write (lunlog,*)' final print:',lprisv 
      write (tmpst,*)' final print:',lprisv 
      lpri=lprisv 
      call xwrite(tmpst,10) 
      if (lpri.ge.5) lpri2=1
      delr=1.e-15
      nlimd=0
      call xstarcalc(lpri2,lnerrd,nlimd,                                &
     &            lpri3,lprid,lunlog,tinf,vturbi,critf,                 &
     &            t,trad,r,delr,xee,xpx,ababs,cfrac,p,lcdd,zeta,        &
     &            mml,mmu,                                              &
     &            epi,ncn2,bremsa,bremsint,                             &
     &            epim,ncn2m,bremsam,                                   &
     &            leveltemp,                                            &
     &            tau0,tauc,                                            &
     &            np2,ncsvn,nlsvn,                                      &
     &            ntotit,                                               &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            elcter,                                               &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &            xilevg,bilevg,rnisg,                                  &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,       &
     &            cabab,opakab,fline,flinel,elin,errc)                
      call heatt(lpri2,lunlog,                                          &
     &       t,r,cfrac,delr,xee,xpx,abel,                               &
     &       epi,ncn2,bremsa,                                           &
     &       leveltemp,                                                 &
     &       ncsvn,nlsvn,                                               &
     &       zrems,zremso,elumab,elumabo,elum,elumo,                    &
     &       rcem,rccemis,opakc,opakcont,cemab,flinel,                  &
     &       brcems)
      lpri2=0
      call stpcut(ldir,lpri2,lunlog,                                    &
     &           ncsvn,nlsvn,                                           &
     &           epi,ncn2,opakc,opakcont,oplin,opakab,delr,             &
     &           dpthc,dpthcont,tau0,tauc) 
      write (lunlog,9902)t,httot,cltot,hmctot 
 9902 format (4(1pe16.8)) 
      write (lunlog,*)' ' 
!                                                                       
!
      call pprint(22,jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,            &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
      nlnprnt=0
      if (lpri.eq.0) nlnprnt=2
      if (lpri.eq.1) nlnprnt=10
      if (lpri.eq.2) nlnprnt=15
      if (lpri.eq.3) nlnprnt=18
      if (lpri.ge.4) nlnprnt=19
      do mm=2,nlnprnt
        call pprint(nlprnt(mm),                                         &
     &                      jkp,trad,xlum,lwri,lpri,r,t,xpx,p,lcdd,     &
     &            numrec,npass,nnmax,nlimd,rmax,xpxcol,xi,zeta,lfix,    &
     &            zremsz,epi,ncn2,abel,cfrac,emult,taumax,xeemin,       &
     &            spectype,specfile,specunit,kmodelname,nloopctl,       &
     &            nparms,parname,partype,parms,parcomm,atcredate,       &
     &            lunlog,tinf,xcol,vturbi,critf,radexp,                 &
     &            delr,rdel,enlum,xee,ababs,                            &
     &            bremsa,bremsint,tau0,dpthc,tauc,                      &
     &            ncsvn,nlsvn,                                          &
     &            ntotit,lnerrd,                                        &
     &            xii,rrrt,pirt,htt,cll,htt2,cll2,httot,cltot,hmctot,   &
     &            cllines,clcont,htcomp,clcomp,clbrems,htfreef,         &
     &            httot2,cltot2,                                        &
     &              xilevg,bilevg,rnisg,                                &
     &            rcem,oplin,rccemis,brcems,opakc,opakcont,cemab,opakab,&
     &            cabab,elumab,elum,zrems,fline,flinel,                 &
     &            zrtmp,zrtmpcol,zrtmph,zrtmpc)                            
         enddo
                
      close(13)                               
!                                                                       
!     Write spectral data file xout_spect1.fits                         
      write(6,*)'xstar: Prepping to write spectral data ' 
      lpril=0
      if (lwri.ge.-1)                                                   &
     & call writespectra(lunlog,lpril,lwri,nparms,parname,partype,parms,&
     &        parcomm,atcredate,t,vturbi,epi,ncn2,dpthc,                &
     &        nlsvn,                                                    &
     &        elin,elum,zrems,zremsz,kmodelname,nloopctl)            
      write (lunlog,*)'after writespectra' 
      if (lwri.ge.0) then 
      lpril=0
      call writespectra2(lunlog,lpril,nparms,parname,partype,parms,  &
     &        parcomm,atcredate,epi,ncn2,dpthc,                         &
     &        np2,nlsvn,                                                &
     &        elum,tau0,kmodelname,nloopctl)                    
      write (lunlog,*)'after writespectra2' 
      lpril=0
      call writespectra3(lunlog,lpril,nparms,parname,partype,parms,  &
     &        parcomm,atcredate,epi,ncn2,dpthc,dpthcont,                &
     &        np2,                                                      &
     &        elum,zrems,zremsz,kmodelname,nloopctl)            
      write (lunlog,*)'after writespectra3' 
      call writespectra4(lunlog,lpril,nparms,parname,partype,parms,  &
     &        parcomm,atcredate,epi,ncn2,dpthc,ababs,                   &
     &                   leveltemp,                                     &
     &        np2,                                                      &
     &        elumab,tauc,kmodelname,nloopctl)                  
      write(6,*)'xstar: Done writing spectral data' 
      endif 
!                                                                       
      call remtms(t2s) 
      ttot=abs(t2s-t1s) 
      write (lunlog,*)'total time',ttot 
      write (tmpst,*)'total time',ttot 
      call xwrite(tmpst,10) 
!                                                                       
      if ((lwri.gt.0).or.(npass.gt.1)) then 
        call fitsclose(lunlog,iunit,istatus) 
        call fitsclose(lunlog,iunit2,istatus) 
        call fitsclose(lunlog,iunit3,istatus) 
        call fitsclose(lunlog,iunit4,istatus) 
        endif
                      
!
      close(unit=lunlog) 
!
      deallocate(masterdata%rdat1)
      deallocate(masterdata%idat1)
      deallocate(masterdata%kdat1)
      deallocate(masterdata%nptrs)
      deallocate(derivedpointers%npar)
      deallocate(derivedpointers%npnxt)
      deallocate(derivedpointers%npfirst)
      deallocate(derivedpointers%npfi)
      deallocate(derivedpointers%npfe)
      deallocate(derivedpointers%nplin)
      deallocate(derivedpointers%nplini)
      deallocate(derivedpointers%npcon)
      deallocate(derivedpointers%npconi)
      deallocate(derivedpointers%npconi2)
      deallocate(derivedpointers%npilev)
      deallocate(derivedpointers%npilevi)
      deallocate(derivedpointers%nlevs)
      deallocate(elum)
      deallocate(elumo)
      deallocate(rcem)
      deallocate(oplin)
      deallocate(fline)
      deallocate(flinel)
      deallocate(tau0)
      deallocate(zrems)
      deallocate(zremso)
      deallocate(zremsz)
      deallocate(dpthc)
      deallocate(dpthcont)
      deallocate(bremsa)
      deallocate(bremsam)
!      write (6,*)'deallocate other data 3'
      deallocate(bremsint)
      deallocate(rccemis)
      deallocate(brcems)
!      write (6,*)'deallocate other data 4'
      deallocate(opakc)
      deallocate(opakcont)
!      write (6,*)'deallocate other data 5'
      deallocate(bilevg)
!      write (6,*)'deallocate other data 6'
      deallocate(rnisg)
!      write (6,*)'deallocate other data 7'
      deallocate(xilevg)
      deallocate(cemab)
      deallocate(cabab)
      deallocate(opakab)
      deallocate(elumab)
      deallocate(elumabo)
      deallocate(tauc)
      deallocate(elin)
      deallocate(errc)
      deallocate(nlin)
      deallocate(zrtmp)
      deallocate(zrtmpcol)
      deallocate(zrtmpc)
      deallocate(zrtmph)
!
      return 
      end                                           

      end module xstar_mod