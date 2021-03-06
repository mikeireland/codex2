"""
Here we prototype a monochromatic calculation of J and F based on an input 
k, r and S.

Original inputs can come from the "PWCan" (i.e. Canberra models from Peter Wood), which 
have a chance of being reproducable. There is also oCet5200fx etc, which have OA07 files
as well. 

The PWCan files have columns:
tau, r, theta ,?, ?, log(rho), ?, ?, v(km/s)


theta is 5040/T, which is multiplied by the excitation energy in eV to get the Boltzmann factor

The M10 was the "P Series" and the M12 the "M Series"
e.g. M12L3P10 started with a 4910 L_sun model:
0.120E+010.491E+040.260E+03
... which matches the start of OA07.M10DM
0.120E+010.491E+040.260E+03

Checking PW versus IA05... thets is similar around tau=1, but 
PW is much too warm in outer layers. I have to start with OA instead.

plt.plot(pwtab['r'], pwtab['theta'])

S10 = planck_mult_const * nus**3/(np.exp(planck_exp_const*pwtab['theta'][10]*nus)-1)

T = 5040/pwtab['theta']

S10_1 = c.h/c.c**2 * (nus*u.Hz)**3/(np.exp(c.h/T[10]/u.K*nus*u.Hz/c.k_B)-1)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
import astropy.io.fits as pyfits
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
import codex2rt
plt.ion()

use_C=True
cubic_interp=False
fits_created = True

def read_and_shorten_pw(fname, maxtau=20, maxr=7.5e13, max_dlrho=0.2, max_dlr=0.1, max_dltau=0.2):
    """
    Read in a Peter Wood file, and shorten it by removing layers.
    """
    dd = np.genfromtxt(fname, skip_header=5, skip_footer=3)
    final_table = []
    #Go through rows 1 at a time
    started=False
    for row in dd:
        if not started:
            if row[1]>maxr:
                continue
            else:
                started=True
        else:
            if row[0]>np.log10(maxtau):
                break
            if (np.log10(final_table[-1][1]/row[1]) < max_dlr) and \
                (row[5]-final_table[-1][5] < max_dlrho) and \
                (row[0]-final_table[-1][0] < max_dltau):
                continue
        final_table += [row]
    final_table = np.array(final_table)
    return {'r':final_table[::-1,1], 'theta':final_table[::-1,2], 'prl':final_table[::-1,3], \
        'lrho':final_table[::-1,5], 'vkms':final_table[::-1,8]}
    
def read_one_tab(file, nth, nprl):
    nrec = nth*nprl
    nread = 0
    data = []
    while nrec - nread > 11:
        line = file.readline()
        data += [float(line[ix:ix+7]) for ix in 7*np.arange(11)]
        nread += 11
    line = file.readline()
    data += [float(line[ix:ix+7]) for ix in 7*np.arange(nrec-nread)]
    data = np.array(data)
    return data.reshape((nprl, nth))
    
def read_opm(fname):
    """Read the OPM file. Format is:
    nth, nprl, title
    initth, dth, initp, dp
    kross table
    mu table
    pel table
    nwave
    (freq wave
    dfreq dwave
    kappa
    sigma) * nwave
    
    The best way to read this is one line at a time!
    """
    with open(fname, "r") as f:
        line = f.readline()
        nth = int(line[:3])
        nprl = int(line[3:6])
        title = line[7:-1]
        line = f.readline()
        ss = line.split()
        th = float(ss[0]) + np.arange(nth)*float(ss[1])
        prl = float(ss[2]) + np.arange(nprl)*float(ss[3])
        line=f.readline()
        kross = read_one_tab(f, nth, nprl)
        line=f.readline()
        mu = read_one_tab(f, nth, nprl)
        line = f.readline()
        pel = read_one_tab(f, nth, nprl)
        line = f.readline()
        nwave = np.abs(int(line))
        nus = np.empty(nwave, dtype=np.float32)
        waves = np.empty(nwave, dtype=np.float32)
        dnus = np.empty(nwave, dtype=np.float32)
        dwaves= np.empty(nwave, dtype=np.float32)
        absn = np.empty((nwave, nprl, nth), dtype=np.float32)
        scat = np.empty((nwave, nprl, nth), dtype=np.float32)
        for i in range(nwave):
            line = f.readline().split()
            nus[i] = line[0]
            waves[i] = line[1]
            line = f.readline().split()
            dnus[i] = line[0]
            dwaves[i] = line[1]
            absn[i] = read_one_tab(f, nth, nprl)
            scat[i] = read_one_tab(f, nth, nprl)
    return th, prl, kross, mu, pel, waves, nus, dnus, absn, scat, title
              
def compute_geometry(rs, central_nmu=5):
    """
    Compute the geometrical terms, namely the delta_s and mu array, plus 
    the weights for J and F calculations (Trapezoidal rule for now) which 
    are in common for all wavelengths.
    
    By Scholz and Wood convention, radii start at the outer layer and go in, but
    this assumes that the radii are in the ascending order here.
    """
    #First, assign arrays with the fast axis going along rays.
    mus = np.zeros((len(rs) + central_nmu, len(rs)))
    
    #Compute p for the central rays
    p = np.empty((len(rs) + central_nmu))  
    mus[:central_nmu, 0] = np.arange(central_nmu,0,-1)/central_nmu
    p[:central_nmu] = rs[0] * np.sqrt(1 - mus[:central_nmu, 0]**2)
    p[central_nmu:] = rs
        
    #Now compute x and mu
    x = np.zeros_like(mus)
    for j in range(0,len(rs)):
        for i in range(0,central_nmu+j+1):
            x[i,j] = np.sqrt(rs[j]**2 - p[i]**2)
            mus[i,j] = x[i,j]/rs[j]

    #For a J integral and F integral, go through the for loop in a
    #different way, indexing the same elements. Maybe this way
    #of indexing the triangle is simpler anyway?
    Jw = np.zeros_like(mus)
    Hw = np.zeros_like(mus)
    for j in range(0,len(rs)):
        for i in range(0,central_nmu+j):
            Jw[i,j] += 0.25*(mus[i,j]-mus[i+1,j])
            Hw[i,j] += 0.25*mus[i,j]*(mus[i,j]-mus[i+1,j])
        for i in range(1,central_nmu+j+1):
            Jw[i,j] += 0.25*(mus[i-1,j]-mus[i,j])
            Hw[i,j] += 0.25*mus[i,j]*(mus[i-1,j]-mus[i,j])
        #As we integrate through the mu=0 region, we have to 
        #take into account only considering this central ray once.
        Jw[central_nmu+j,j] *= 2
        
            
    #Finally, compute delta_s
    delta_s = x[:-1,1:] - x[:-1,:-1]
    return delta_s, Jw, Hw, mus
    
def compute_grid(delta_s, k_r, S_C1, S_C2, expdy):
    """
    For x single wavelength, compute the grid of \Delta I coefficients
    going in the positive and negative directions.
    """
    #Firstly, compute the optical depths, which we call y here just because 
    #this was the integration variable chosen in rtransfer_notes. Usually 
    #called t, which is also confusing!
    #The following arrays can be empty, but we want to debug...
    central_nmu = delta_s.shape[0]-delta_s.shape[1]
    y = np.zeros_like(delta_s) 
    
    for i in range(y.shape[0]):
        for j in range(np.maximum(i-central_nmu,0),y.shape[1]):
            y[i,j] = delta_s[i,j]*0.5*(k_r[j] + k_r[j+1])
            
    #We need not take exp(0) but can here for simplicity
    used = y != 0
    expdy[used] = np.exp(-y[used])
    S_C1[used] = (y[used] - 1 + expdy[used])/y[used]
    S_C2[used] = (1 - (1 + y[used])*expdy[used])/y[used]

def compute_rtransfer(S_C1, S_C2, expdy, Jw, Hw, Jmat, Hmat):
    """
    Compute the radiative transfer matrices for J and H. There are 5 input
    matrices:
    
    Parameters
    ----------
    S_C1:
        (np-1, nr-1) array. 1st coefficients of the source function
    S_C2:
        
    """
    #Firstly, compute the optical depths, which we call y here just because 
    #this was the integration variable chosen in rtransfer_notes. Usually 
    #called t, which is also confusing!
    #The following arrays can be empty, but we want to debug...
    nr = Jw.shape[1]
    central_nmu = Jw.shape[0]-Jw.shape[1]
    Iinner_C = np.zeros((central_nmu+nr, nr))
    
    #We have to do our first 3-dimensional loop here, considering inwards going rays
    #first (mu <= 0), then outwards going rays (mu>0)
    for j in range(0,nr):
        #i is the y value.
        for i in range(0,central_nmu+j+1):
            #Integrating outwards along the incoming ray.
            exptau = 1
            for k in range(0,nr-j-1):
                Jmat[j,j+k] += exptau*S_C1[i,j+k]*Jw[i,j]
                Jmat[j,j+k+1] += exptau*S_C2[i,j+k]*Jw[i,j]
                Hmat[j,j+k] -= exptau*S_C1[i,j+k]*Hw[i,j]
                Hmat[j,j+k+1] -= exptau*S_C2[i,j+k]*Hw[i,j]
                #Inner Boundary and mu=0 rays
                if (j==0) or (i==central_nmu+j):
                    Iinner_C[i,j+k] += exptau*S_C1[i,j+k]
                    Iinner_C[i,j+k+1] += exptau*S_C2[i,j+k]
                exptau *= expdy[i,j+k]
        
    #We have to do our first 3-dimensional loop here, considering inwards going rays
    #first, then outwards going rays.
    for j in range(0,nr):
        #i is the y value. We have already covered the mu=0 ray, so don't
        #need to consider it again.
        for i in range(0,central_nmu+j):
            #Integrating inwards along the outgoing ray.
            exptau = 1
            for k in range(0,j):
                Jmat[j,j-k] += exptau*S_C1[i,j-k-1]*Jw[i,j]
                Jmat[j,j-k-1] += exptau*S_C2[i,j-k-1]*Jw[i,j]
                Hmat[j,j-k] += exptau*S_C1[i,j-k-1]*Hw[i,j]
                Hmat[j,j-k-1] += exptau*S_C2[i,j-k-1]*Hw[i,j]
                exptau *= expdy[i,j-k-1]
            #Now we're at the inner boundary, or about to go outwards, 
            #and have to apply that boundary condition. 
            if i < central_nmu:
                Jmat[j,0] += 2*exptau*Jw[i,j]
                Hmat[j,0] += 2*exptau*Hw[i,j]
                for k in range(0,nr):
                    Jmat[j,k] -= exptau*Iinner_C[i,k]*Jw[i,j]
                    Hmat[j,k] -= exptau*Iinner_C[i,k]*Hw[i,j]
            else:
                for k in range(i-central_nmu,nr):
                    Jmat[j,k] += exptau*Iinner_C[i,k]*Jw[i,j]
                    Hmat[j,k] += exptau*Iinner_C[i,k]*Hw[i,j]

def calc_J_F(theta, lrho, prl, th_grid, prl_grid,delta_s, Jw, Hw, absn, nus, dnus):
    """
    Caculate J and F as a function or R.
    
    Issues - this should *not* get both rho and prl from 
    the Wood model.
    
    Parameters
    ----------
    theta : numpy array
        5040/T
    prl : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    delta_s : TYPE
        DESCRIPTION.
    Jw : TYPE
        DESCRIPTION.
    Hw : TYPE
        DESCRIPTION.
    absn : TYPE
        DESCRIPTION.
    nus : TYPE
        DESCRIPTION.
    dnus : TYPE
        DESCRIPTION.

    Returns
    -------
    J and F as a function of R

    """
    nr = len(theta)
    nnu = len(nus)
    Js = np.empty((nnu, nr))
    Hs = np.empty((nnu, nr))
    Jkappa = np.zeros((nr))
    Bkappa = np.zeros((nr))
    Jmat = np.require(np.zeros( (nr,nr) ), requirements=['C'])
    Hmat = np.require(np.zeros( (nr,nr) ), requirements=['C'])
    
    #Pre-compute constants.
    planck_exp_const = 6.626e-34/1.38e-23/5040
    planck_mult_const = 2*6.626e-34/3e8**2

    #Pre-compute interpolation weights for linear interpolation.
    prl_ix = np.interp(prl, prl_grid, np.arange(len(prl_grid,)))
    th_ix = np.interp(theta, th_grid, np.arange(len(th_grid,)))
    prl_ix = np.minimum(prl_ix, len(prl_grid)-1.001)
    th_ix = np.minimum(th_ix, len(th_grid)-1.001)
    prl_int = prl_ix.astype(int)
    th_int = th_ix.astype(int)
    prl_frac = prl_ix - prl_int
    th_frac = th_ix - th_int
    kappa_r = np.empty(nr)
    xvect = np.array([1-prl_frac,prl_frac]).T
    yvect = np.array([1-th_frac,th_frac]).T
    
    for w_ix in range(nnu):
        #Compute opacity
        ab = absn[w_ix]
        A = np.array([[ab[prl_int,th_int],ab[prl_int,th_int+1]],[ab[prl_int+1,th_int],ab[prl_int+1,th_int+1]]])
        kappa_r = np.sum(np.einsum('ij,jki->ik', xvect, A)*yvect, axis=1)
        k_r = 10**(kappa_r + lrho)
        
        #Compute the initial source function fot these layers
        source_fns = planck_mult_const * nus[w_ix]**3/(np.exp(planck_exp_const*theta*nus[w_ix])-1)
        
        #We have everything we need! Lets compute J and H, in three steps.
        if use_C:
            codex2rt.compute_grid(delta_s, k_r, S_C1, S_C2, expdy)
        else:
            compute_grid(delta_s, k_r, S_C1, S_C2, expdy)
        #Re-zero the initial matrices. Could be done in the C code...
        Hmat[:]=0
        Jmat[:]=0
        if use_C:
            codex2rt.compute_rtransfer(S_C1, S_C2, expdy, Jw, Hw, Jmat, Hmat)
        else:
            compute_rtransfer(S_C1, S_C2, expdy, Jw, Hw, Jmat, Hmat)
        Js[w_ix] = np.dot(Jmat, source_fns)
        Hs[w_ix] = np.dot(Hmat, source_fns)
        Jkappa += kappa_r*Js[w_ix]*dnus[w_ix]
        #!!! To be changed with scattering.
        Bkappa += kappa_r*source_fns*dnus[w_ix] 
    #Compute the total flux
    Ft = np.zeros((nr))
    Jt = np.zeros((nr))
    for i in range(nr):
        Ft[i] = 4*np.pi*np.sum(Hs[:,i]*dnus)
        Jt[i] = np.sum(Js[:,i]*dnus)
    return Ft, Jkappa, Bkappa

def req_solve(theta, Ltarg, rs_m, lrho, prl, th_grid, prl_grid,delta_s, Jw, Hw, absn, nus, dnus):  
    """
    Solve for Radiative Equilibrium. We want both L_const and J=B to be true
    How to make this possible? Maybe it isn't, just like the intuition 
    in the Scholz days.

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    Ltarg : TYPE
        DESCRIPTION.
    rs_m : TYPE
        DESCRIPTION.
    Others: See calc_J_F
    
    
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    L_sun = 3.828e+26 #SI units
    Ft, Jkappa, Bkappa = calc_J_F(theta, lrho, prl, th_grid, prl_grid,delta_s, Jw, Hw, absn, nus, dnus)
    scaled_L_error = (Ft*4*np.pi*rs_m**2/L_sun) / Ltarg - 1
    scaled_J_error = (Jkappa - Bkappa)/Bkappa
    print("Surface scaled L and J errors: {:.2f} {:.2f}".format(scaled_L_error[-1], scaled_J_error[-1]))
    scaled_L_error[-4:] = scaled_J_error[-4:] 
    #return scaled_L_error*scaled_J_error**2 + scaled_J_error*scaled_L_error**2
    return scaled_L_error

if __name__=="__main__":
    fits_fname = 'OPM00_os4300_0212.fits'
    #Read in the dynamical model
    pwfile = '/Users/mireland/theory/IScommon/dustmmi/pw/PWCan.M12L3P10'
    pwtab = read_and_shorten_pw(pwfile)
    Ltarg = 0.491E+04
    
    #This is everything we need for geometry
    delta_s, Jw, Hw, mus = compute_geometry(pwtab['r'])
    #Require that key arrays are contiguous.
    Jw = np.require(Jw, requirements=['C'])
    Hw = np.require(Hw, requirements=['C'])

    #The next code turns this into a fits file which will read quicker
    if fits_created:
        ff = pyfits.open(fits_fname)
        absn = ff[0].data
        th = ff[0].header['CRPIX1'] + ff[0].header['CDELT1']*np.arange(absn.shape[2])
        prl = ff[0].header['CRPIX2'] + ff[0].header['CDELT2']*np.arange(absn.shape[1])
        nus = ff[4].data['nu']
        waves = ff[4].data['wave']
        dnus = ff[4].data['dnus']
    else:
        #Read in the opacity file
        opmfile = '/Users/mireland/code/scholz_code/tables/OPM00_os4300_0212'
        th, prl, kross, mu, pel, waves, nus, dnus, absn, scat, title = read_opm(opmfile)
        hdu1 = pyfits.PrimaryHDU(absn)
        hdu1.header['CRPIX1'] = (th[0], 'Theta in 1/eV (5040/T)')
        hdu1.header['CDELT1'] = (th[1]-th[0], 'Delta theta in 1/eV')
        hdu1.header['CRPIX2'] = (prl[0], 'Pressure logarithm start (cgs)')
        hdu1.header['CDELT2'] = (prl[1]-prl[0], 'Delta pressure logarithm')
        hdu2 = pyfits.ImageHDU(scat)
        hdu3 = pyfits.ImageHDU(kross)
        hdu4 = pyfits.ImageHDU(mu)
        col1 = pyfits.Column(name='wave', format='E', array=waves)
        col2 = pyfits.Column(name='nu', format='E', array=nus)
        col3 = pyfits.Column(name='dnus', format='E', array=dnus)
        hdu5 = pyfits.BinTableHDU.from_columns(pyfits.ColDefs([col1, col2, col3]))
        hdulist = pyfits.HDUList([hdu1,hdu2,hdu3,hdu4,hdu5])
        hdulist.writeto(fits_fname)
            
    #waves[1500] is 1.7 microns and a good wavelength to try radiative transfer.
    #Now we find opacity k from rho and kappa.
    w_ixs = np.arange(len(nus)) #[1500]#np.arange(500) #np.arange(len(nus)) #np.range(100) #np.range(len(nus))
    S_C1 = np.require(np.zeros_like(delta_s), requirements=['C'])
    S_C2 = np.require(np.zeros_like(delta_s), requirements=['C'])
    expdy =np.require(np.zeros_like(delta_s), requirements=['C'])
    
    if False:
        nr = len(pwtab['prl'])
        Js = np.empty((len(w_ixs), nr))
        Hs = np.empty((len(w_ixs), nr))
        Jmat = np.require(np.zeros( (nr,nr) ), requirements=['C'])
        Hmat = np.require(np.zeros( (nr,nr) ), requirements=['C'])
        
        #Pre-compute constants.
        planck_exp_const = 6.626e-34/1.38e-23/5040
        planck_mult_const = 2*6.626e-34/3e8**2
    
        #Pre-compute interpolation weights for linear interpolation.
        prl_ix = np.interp(pwtab['prl'], prl, np.arange(len(prl)))
        th_ix = np.interp(pwtab['theta'], th, np.arange(len(th)))
        prl_int = prl_ix.astype(int)
        th_int = th_ix.astype(int)
        prl_frac = prl_ix - prl_int
        th_frac = th_ix - th_int
        kappa_r = np.empty(nr)
        xvect = np.array([1-prl_frac,prl_frac]).T
        yvect = np.array([1-th_frac,th_frac]).T
        
        start = time.time()
        tinterp = 0
        for ix, w_ix in enumerate(w_ixs): 
            #Strangely, this is quite slow - mostly extracting from absn
            #Interpolation: 133ms
            #rtransfer: 222ms
            #dot: 17ms
            #compute_grid: 22ms
            #Everything else: 80ms
            #Going any further needs putting this full loop in C.
            #t1 = time.time()
            if cubic_interp:
                absn_func = RectBivariateSpline(prl, th, absn[w_ix])
                
                #Now interpolate for our layers
                kappa_r = absn_func(pwtab['prl'],pwtab['theta'],grid=False)
            else:
              ab = absn[w_ix]
              A = np.array([[ab[prl_int,th_int],ab[prl_int,th_int+1]],[ab[prl_int+1,th_int],ab[prl_int+1,th_int+1]]])
              kappa_r = np.sum(np.einsum('ij,jki->ik', xvect, A)*yvect, axis=1)
            #t2=time.time()
            #tinterp += t2-t1
            
            #pdb.set_trace()
            #kappa_r = absn[w_ix][prl_int:prl_int+1, th_int:th_int+1] 
            k_r = 10**(kappa_r + pwtab['lrho'])
            
            #Compute the initial source function fot these layers
            source_fns = planck_mult_const * nus[w_ix]**3/(np.exp(planck_exp_const*pwtab['theta']*nus[w_ix])-1)
            
            #We have everything we need! Lets compute J and H, in three steps.
            if use_C:
                codex2rt.compute_grid(delta_s, k_r, S_C1, S_C2, expdy)
            else:
                compute_grid(delta_s, k_r, S_C1, S_C2, expdy)
            #Re-zero the initial matrices. Could be done in the C code...
            Hmat[:]=0
            Jmat[:]=0
            if use_C:
                codex2rt.compute_rtransfer(S_C1, S_C2, expdy, Jw, Hw, Jmat, Hmat)
            else:
                compute_rtransfer(S_C1, S_C2, expdy, Jw, Hw, Jmat, Hmat)
            Js[ix] = np.dot(Jmat, source_fns)
            Hs[ix] = np.dot(Hmat, source_fns)
        print("Total Time (ms): {:.1f}".format((time.time()-start)*1e3))
        print("Interpolation Time (ms): {:.1f}".format(tinterp*1e3))
        
        #Compute the total flux
        Ft = np.zeros((nr))
        for i in range(nr):
            Ft[i] = 4*np.pi*np.sum(Hs[:,i]*dnus[w_ixs])
            
    theta = pwtab['theta']
    Ft, Jkappa, Bkappa = calc_J_F(theta, pwtab['lrho'], pwtab['prl'], th, prl,delta_s, Jw, Hw, absn, nus, dnus)
    #plt.clf()
    plt.plot(((Ft*u.W/u.m**2)*4*np.pi*(pwtab['r']*u.cm)**2).to(u.L_sun).value)
    plt.xlabel('Layer')
    plt.ylabel('Luminosity (L_sun)')
    flux = Ft[10]*u.W/u.m**2
    print( "Rough Teff: ", ((flux/c.sigma_sb)**(1/4)).si )
    
    #Now legs try to solve!(theta, Ltarg, rs_m, lrho, prl, th_grid, prl_grid,delta_s, Jw, Hw, absn, nus, dnus):
    soln = root(req_solve, theta, args=(Ltarg, 0.01*pwtab['r'], pwtab['lrho'], pwtab['prl'], th, prl, delta_s, Jw, Hw, absn, nus, dnus))
    
    #To Check against a reference calculation...
    #np.save('Jmat.npy', Jmat)
    #np.save('Hmat.npy', Hmat)
    #Jmat_old =np.load('Jmat.npy')
    #Hmat_old =np.load('Hmat.npy')
    