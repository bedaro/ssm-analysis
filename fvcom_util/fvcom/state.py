from dataclasses import dataclass
from scipy.io import FortranFile
import numpy as np

from .grid import FvcomGrid
from .depth import DepthCoordinate

@dataclass
class FvcomState:
    dcoord: DepthCoordinate
    grid: FvcomGrid
    iint: int
    u: np.array
    v: np.array
    w: np.array
    tke: np.array
    teps: np.array
    q2: np.array
    q2l: np.array
    l: np.array
    s: np.array
    t: np.array
    rho: np.array
    tmean: np.array
    smean: np.array
    rmean: np.array
    s1: np.array
    t1: np.array
    rho1: np.array
    tmean1: np.array
    smean1: np.array
    rmean1: np.array
    km: np.array
    kh: np.array
    kq: np.array
    ua: np.array
    va: np.array
    el1: np.array
    et1: np.array
    h1: np.array
    d1: np.array
    dt1: np.array
    rtp: np.array
    el: np.array
    et: np.array
    h: np.array
    d: np.array
    dt: np.array
    el_eqi: np.array
    el_atmo: np.array
    wqm: np.array
    dye: np.array
    dymean: np.array
    gotm: bool = False  # Whether the general ocean turbulence model is in use
    equi_tide: bool = False
    atmo_tide: bool = False
    water_quality: bool = False
    dye_release: bool = False

    def to_restart_file(self, out_file):
        with FortranFile(out_file, 'w') as f:
            f.write_record([self.iint])

            f.write_record(self.u)
            f.write_record(self.v)
            f.write_record(self.w)
            if self.gotm:
                f.write_record(self.tke)
                f.write_record(self.teps)
            else:
                f.write_record(self.q2)
                f.write_record(self.q2l)
                f.write_record(self.l)

            f.write_record(self.s.flatten(order='F'))
            f.write_record(self.t.flatten(order='F'))
            f.write_record(self.rho.flatten(order='F'))
            f.write_record(self.tmean.flatten(order='F'))
            f.write_record(self.smean.flatten(order='F'))
            f.write_record(self.rmean.flatten(order='F'))
    
            f.write_record(self.s1.flatten(order='F'))
            f.write_record(self.t1.flatten(order='F'))
            f.write_record(self.rho1.flatten(order='F'))
            f.write_record(self.tmean1.flatten(order='F'))
            f.write_record(self.smean1.flatten(order='F'))
            f.write_record(self.rmean1.flatten(order='F'))

            f.write_record(self.km)
            f.write_record(self.kh)
            f.write_record(self.kq)
            f.write_record(self.ua)
            f.write_record(self.va)
            f.write_record(self.el1)
            f.write_record(self.et1)
            f.write_record(self.h1)
            f.write_record(self.d1)
            f.write_record(self.dt1)
            f.write_record(self.rtp)

            f.write_record(self.el)
            f.write_record(self.et)
            f.write_record(self.h)
            f.write_record(self.d)
            f.write_record(self.dt)

            if self.equi_tide:
                f.write_record(self.el_eqi)
            if self.atmo_tide:
                f.write_record(self.el_atmo)
            if self.water_quality:
                f.write_record(self.wqm)
            if self.dye_release:
                f.write_record(self.dye)
                f.write_record(self.dyemean)

    def dens2(self):
        """Based on FVCOM 2.7 dens2.F for the case where CTRL_DENS = sigma-t

        Builds a new node density field based on the temperature/salinity
        fields and the equation of state. DOES NOT HANDLE WET/DRY CASE
        """
        tfld = self.t1[:-1]
        sfld = self.s1[:-1]
        rhof = 6.76786136E-6 * sfld ** 3 - 4.8249614E-4 * sfld ** 2 + 8.14876577E-1 * sfld - 0.22584586
        rhof *= 1.667E-8 * tfld ** 3 - 8.164E-7 * tfld ** 2 + 1.803E-5 * tfld
        rhof += 1 - 1.0843E-6 * tfld ** 3 + 9.8185E-5 * tfld ** 2 - 4.786E-3 * tfld
        rhof *= 6.76786136E-6 * sfld ** 3 - 4.8249614E-4 * sfld ** 2 + 8.14876577E-1 * sfld + 3.895414E-2
        rhof = rhof - (tfld - 3.98) ** 2 * (tfld + 283) / (503.57 * (tfld + 67.26))
        # Does not handle wet/dry case!
        # Put the final row of 0's back for the last layer
        rho1 = (np.append(rhof, [self.t1[-1]], axis=0) * 1e-3).astype(np.float32)
        self.rho1 = rho1

    def n2e3d(self, nvar):
        """Based on FVCOM 2.7 subroutine N2E3D from utilities.F.

        Averages node properties to set the element center
        """
        evar = np.zeros((self.dcoord.kb, self.grid.n+1), dtype=np.float32)
        for i in range(self.grid.n):
            # Get the node indices at each vertex of this element
            n1, n2, n3 = [self.grid.nv[j,i] for j in range(3)]
            evar[:,i+1] = np.mean(nvar[:,[n1-1,n2-1,n3-1]], axis=1)
        return evar

    def __setattr__(self, name, value):
        """Automatically update density and/or element fields when node fields change"""
        super().__setattr__(name, value)
        if not hasattr(self, 'initialized'):
            return
        if name == "t1":
            self.dens2()
            self.t = self.n2e3d(value)
        elif name == "s1":
            self.dens2()
            self.s = self.n2e3d(value)
        elif name == "rho1":
            self.rho = self.n2e3d(value)

    def __post_init__(self):
        self.initialized = True

    @staticmethod
    def read_restart_file(restart_file, grid, dcoord, gotm=False, equi_tide=False,
            atmo_tide=False, water_quality=False, dye_release=False):
        """Read a FVCOM 2.7 restart file.

        Currently only temperature, salinity, and density arrays are reshaped.
        The rest are read in flat.
        """
        kb = dcoord.kb
        with FortranFile(restart_file, 'r') as f:
            iint, = f.read_ints(np.int32)

            non = np.array([])

            u = f.read_reals(np.float32)
            v = f.read_reals(np.float32)
            w = f.read_reals(np.float32)
            if gotm:
                tke = f.read_reals(np.float32)
                teps = f.read_reals(np.float32)
                q2 = non
                q2l = non
                l = non
            else:
                tke = non
                teps = non
                q2 = f.read_reals(np.float32)
                q2l = f.read_reals(np.float32)
                l = f.read_reals(np.float32)

            s = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')
            t = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')
            rho = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')
            tmean = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')
            smean = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')
            rmean = f.read_reals(np.float32).reshape(kb, grid.n+1, order='F')

            s1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')
            t1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')
            rho1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')
            tmean1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')
            smean1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')
            rmean1 = f.read_reals(np.float32).reshape(kb, grid.m, order='F')

            km = f.read_reals(np.float32)
            kh = f.read_reals(np.float32)
            kq = f.read_reals(np.float32)
            ua = f.read_reals(np.float32)
            va = f.read_reals(np.float32)
            el1 = f.read_reals(np.float32)
            et1 = f.read_reals(np.float32)
            h1 = f.read_reals(np.float32)
            d1 = f.read_reals(np.float32)
            dt1 = f.read_reals(np.float32)
            rtp = f.read_reals(np.float32)

            el = f.read_reals(np.float32)
            et = f.read_reals(np.float32)
            h = f.read_reals(np.float32)
            d = f.read_reals(np.float32)
            dt = f.read_reals(np.float32)

            if equi_tide:
                el_eqi = f.read_reals(np.float32)
            else:
                el_eqi = non
            if atmo_tide:
                el_atmo = f.read_reals(np.float32)
            else:
                el_atmo = non
            if water_quality:
                wqm = f.read_reals(np.float32)
            else:
                wqm = non
            if dye_release:
                dye = f.read_reals(np.float32)
                dymean = f.read_reals(np.float32)
            else:
                dye = non
                dymean = non

        return FvcomState(dcoord, grid, iint, u, v, w, tke, teps, q2, q2l, l, s,
                t, rho, tmean, smean, rmean, s1, t1, rho1, tmean1, smean1,
                rmean1, km, kh, kq, ua, va, el1, et1, h1, d1, dt1, rtp, el,
                et, h, d, dt, el_eqi, el_atmo, wqm, dye, dymean, gotm=gotm,
                equi_tide=equi_tide, atmo_tide=atmo_tide,
                water_quality=water_quality, dye_release=dye_release)
