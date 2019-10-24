from .emlines import EmissionSpectrum
from .SPError import SPError

import numpy as np
import matplotlib.pyplot as mpl
import h5py
from sedpy import observate

import time

try:
    import fsps
except ImportError:
    raise SPError("Package fsps not found, please install this prior to using this package.")




class SPSModel:

    DefaultBands = ['wfc_acs_f435w','wfc_acs_f606w','wfc_acs_f814w',\
                    'wfc3_ir_f105w','wfc3_ir_f125w','wfc3_ir_f140w','wfc3_ir_f160w']

    DefaultColors = [('wfc3_ir_f125w','wfc3_ir_f160w'),('wfc_acs_f606w','wfc_acs_f814w')]

    def __init__(self,bands=None,**kwargs):
        self.model = fsps.StellarPopulation(add_neb_emission=False,**kwargs)

        ## NEED TO UPDATE THIS TO DEAL WITH INOUT KWARGS OVERRIDE DEFAULTS
        self.model.params['imf_type'] = 1   # Chabrier IMF
        self.model.params['sfh'] = 1  # Tau model SFH
        self.model.params['logzsol'] = 0 # solar metallicity
        self.model.params['dust_type'] = 2 # Calzetti2000
        self.model.params['dust1'] = 1.0 # must be 1 for Calzetti2000
        self.model.params['dust2'] = 0.00 * 4.05 # E(B-V) * RV

        if bands is None:
            self.set_bands(SPSModel.DefaultBands)
        else:
            self.filter_names = None

        return None

    def set_bands(self,filter_names):
        for name in filter_names:
            if not name in fsps.list_filters():
                raise SPError(f"{name} not available in current version of FSPS. Please choose one of {fsps.list_filters()}")
            else:
                continue

        self.filter_names = filter_names
        return None


    def model_ditribution(self,sps,t_age,z_err,c_err,n_points,color_idx,names):
        if self.filter_names is None:
            raise SPError("Must first define the filters with set_bands method.")

        zs_data = np.zeros(n_points)
        cs_data = np.zeros(n_points)

        for i in range(n_points):
            z_real = np.random.uniform(1,3.0)
            z_phot = np.random.normal(z_real,z_err)

            magsLines = self.model.get_mags(tage=t_age,redshift=z_real, bands=self.filter_names)
            color_real = magsLines[color_idx[0]] - magsLines[color_idx[1]]
            color_phot = np.random.normal(color_real,c_err)

            zs_data[i] = z_phot
            cs_data[i] = color_phot
        return zs_data,cs_data


    def _define_colors(self,colorList=None):
        color_idxs = []
        if colorList is  None:
            assumedColors = SPSModel.DefaultColors
        else:
            assumedColors = colorList

        for clr in assumedColors:
            color_idx = (self.filter_names.index(clr[0]),self.filter_names.index(clr[1]))
            color_idxs.append(color_idx)
        return color_idxs


    def added_emission_line_spectra(self,age,ebv,HaEw,logzsol=0,**kwargs):
        wave, spec = self.model.get_spectrum(tage=age,peraa=True)
        emLineSpec = EmissionSpectrum(wave)
        emLineSpec.halpha_model(spec, HaEw, logzsol=logzsol,**kwargs)
        return wave, spec+emLineSpec.flux

    def get_colors_emission_line(self,zGrid,age,ebv,HaEw,colorList=None,logzsol=0,**kwargs):
        FilterList = observate.load_filters(self.filter_names)
        color_idxs = self._define_colors(colorList)
        WaveModel,SpecModel = self.added_emission_line_spectra(age,ebv,HaEw,logzsol=logzsol,**kwargs)

        ncolors = len(color_idxs)
        nzGrid = len(zGrid)
        color_grid = np.zeros([ncolors,nzGrid])
        for i,z in enumerate(zGrid):
            mags = observate.getSED(WaveModel*(1+z),SpecModel, filterlist=FilterList)
            colors = [ mags[c[0]]-mags[c[1]] for c in color_idxs ]
            color_grid[:,i] = colors
        return color_grid

    def create_grid(self,fname,ebv=None,ages=None,HaEW=None,zGrid=None, colorList=None,debug=False):

        def _write_grid(fname):
            fout = h5py.File(fname,"w")

            gridDataset = fout.create_dataset("grid",data=model_grid)
            fout.create_dataset("ages",data=ages)
            fout.create_dataset("zgrid",data=zGrid)
            fout.create_dataset("dust",data=ebv)
            fout.create_dataset("Ha",data=HaEW)

            gridDataset.attrs["shape"] = model_grid.shape
            gridDataset.attrs["colors"] = color_idxs
            gridDataset.attrs["AXIS0"] = "colors"
            gridDataset.attrs["AXIS1"] = "Ha"
            gridDataset.attrs["AXIS2"] = "dust"
            gridDataset.attrs["AXIS3"] = "ages"
            gridDataset.attrs["AXIS4"] = "zgrid"
            fout.close()
            return None

        color_idxs = self._define_colors(colorList)

        if debug is True:
            print("CLRS",color_idxs)
        if zGrid is None:
            zGrid = np.linspace(1,3,50)

        if ebv is None:
            ebv = [0,0.1,0.2,0.3]

        if ages is None:
            ages = np.logspace(-3,np.log10(2),50)

        if HaEW is None:
            HaEW = np.linspace(0,200,100)


        FilterList = observate.load_filters(self.filter_names)


        ncolors = len(color_idxs)
        nAges = len(ages)
        nHa = len(HaEW)
        ndust = len(ebv)
        nzGrid = len(zGrid)

        model_grid = np.zeros([ncolors,nHa,ndust,nAges,nzGrid])

        if debug is True:
            print("Start creating grid")
        tstart = time.time()
        for i,value in enumerate(ebv):
            if debug is True:
                print(i)
            self.model.params['dust2'] =  value * 4.05 # E(B-V) * RV
            for j,age in enumerate(ages):
                wave, spec = self.model.get_spectrum(tage=age,peraa=True)
                emLineSpec = EmissionSpectrum(wave)
                for k,Ha in enumerate(HaEW):
                    emLineSpec.halpha_model(spec, Ha, logzsol=0)
                    for l,z in enumerate(zGrid):
                        mags = observate.getSED(wave*(1+z), spec+emLineSpec.flux, filterlist=FilterList)
                        colors = [ mags[c[0]]-mags[c[1]] for c in color_idxs ]

                        model_grid[:,k,i,j,l] = colors
        tend = time.time()
        if debug is True:
            print(f"Elapsed {tend-tstart} secs on grid creation")
        _write_grid(fname)

        # self.model.params['dust2'] = 0.00 * 4.05 # E(B-V) * RV
        # mags= self.model.get_mags(tage=ages[0],redshift=zGrid[0], bands=self.filter_names)
        # print(mags[:-1]-mags[1:])
        #
        # wave, spec = self.model.get_spectrum(tage=ages[0],peraa=True)
        # emLineSpec = EmissionSpectrum(wave)
        # emLineSpec.halpha_model(spec, HaEW[1], logzsol=0)
        # mags = observate.getSED(wave*(1+zGrid[0]), spec, filterlist=FilterList)
        # print(mags[:-1]-mags[1:])
        #
        #
        # selLims = (2900,9600)
        # fig = mpl.figure(figsize=(12,8))
        # ax = fig.add_axes([0.05,0.05,0.9,0.6])
        # axLines = fig.add_axes([0.05,0.65,0.9,0.3],sharex=ax)
        # emLineSpec.plot_spectrum(ax=axLines,color="LimeGreen")
        # selection = (wave>selLims[0]) & (wave<selLims[1])
        # ax.plot(wave[selection],spec[selection]+emLineSpec.flux[selection],color="LimeGreen")
        # ax.plot(wave[selection],spec[selection],color="k")
        # ax.set_xlim(selLims[0],selLims[1])
        # axLines.tick_params(labelbottom=False)
        # mpl.show()

        #
        # self.model.params['dust2'] = 0.20 * 4.05 # E(B-V) * RV
        # print(self.model.get_mags(tage=0.1,redshift=2.0, bands=self.filter_names))
        return




def running_median(var_x,var_y,nbins,bin_width=None,pre_selx=None,pre_sely=None):

    x_center = np.linspace(min(var_x),max(var_x),nbins)

    if bin_width is None:
        bin_width = 1.5*(x_center[1]-x_center[0])

    percentiles = np.zeros([x_center.size,3])
    for i in range(nbins):
        bin_sel = (var_x>=x_center[i]-bin_width)*(var_x<x_center[i]+bin_width)
        binmed = x_center[i]


        if pre_selx is None and pre_sely is None:
            V=var_y[bin_sel]
        elif pre_selx is None:
            V=var_y[bin_sel*pre_sely]
        elif pre_sely is None:
            V=var_y[pre_selx*bin_sel]
        else:
            V=var_y[pre_selx*bin_sel*pre_sely]

        if np.size(V)>0:
            percentiles[i,:] = np.percentile(V,[16,50,84])
        else:
            percentiles[i,:] = [0,0,0]

    return x_center,percentiles


class GalaxyData:

    def __init__(self,table):
        self.table = table
        return
