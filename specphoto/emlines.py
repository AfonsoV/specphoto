from .SPError import SPError
from .dust import klambda_salim_highz

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as mpl


strFolder = "/".join(__file__.split("/")[:-1])
__folder__ = f"/{strFolder}"

class EmissionLine:

    def __init__(self,line_center, amp=1,fwhm=1):
        self.lambda_c = line_center
        self.amplitude = amp
        self.fwhm = fwhm
        self.sigma = self._fhwm2sigma(fwhm)

    def _fhwm2sigma(self,fwhm):
        return fwhm / (2*np.sqrt(2*np.log(2)))

    def _gaussian(self,x):
        return self.amplitude * np.exp(-(x-self.lambda_c)**2/(2*self.sigma**2))

    def spectrum(self,wavelength):
        return self._gaussian(wavelength)



class EmissionSpectrum:

    def __init__(self, wavelength):
        self.wave = wavelength
        self.flux = np.zeros_like(wavelength)
        self.lines = []
        self.line_data = Table.read(f"{__folder__}/data/anders2003.spec",format="ascii")
        return None

    def _compute_weights(self,zmet):
        zTable = np.asarray([0.0004,0.004,0.008])
        zSol = np.log10(zTable/0.02)

        if zmet>zSol[-1]:
            return [0,0,1]
        elif zmet<zSol[0]:
            return [1,0,0]
        else:
            i = np.where(zmet>zSol)[0][-1]
            total = np.abs(zSol[i]-zSol[i+1])

            w = [0,0,0]
            w[i] = np.abs(zmet-zSol[i])/total
            w[i+1] =np.abs(zmet-zSol[i+1])/total
            return w


    def add_emission_line(self,line_wave,amp=1,fwhm=10):
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        emline = EmissionLine(line_wave,amp,sigma)
        lineSpec = emline.spectrum(self.wave)
        self.flux += lineSpec
        self.lines.append(emline)
        return emline

    def plot_spectrum(self,ax=None,**kwargs):
        if ax is None:
            fig,ax = mpl.subplots()
            ax.plot(self.wave,self.flux,**kwargs)
        else:
            ax.plot(self.wave,self.flux,**kwargs)
        return ax

    def _test_model(self,sps,age=0.5,fwhm=10,window_continuum=20):
        wavelength, f_nu = sps.get_spectrum(tage=age)
        lineCenter = 6564.5

        continuum_selection = (self.wave>lineCenter-window_continuum) & (self.wave<lineCenter+window_continuum)
        line_selection = (self.wave>lineCenter-5) & (self.wave<lineCenter+5)
        region = continuum_selection ^ line_selection
        continuum = np.nanmedian(f_nu[region])

        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        lineAmp = 50 * continuum / (sigma * np.sqrt(2*np.pi))
        emLine = self.add_emission_line(f_nu,lineCenter,lineAmp,fwhm=fwhm)

        fig,ax = mpl.subplots()
        window = 150
        selection = (wavelength>lineCenter-window) & (wavelength<lineCenter+window)
        ax.plot(wavelength[selection],(f_nu+self.flux)[selection],"-",color="red")
        ax.plot(wavelength[selection],f_nu[selection],"-",color="black")

    def _get_amplitude_from_ew(self,f_nu,lineCenter,ew,sigma,window_continuum=20):
        continuum_selection = (self.wave>lineCenter-window_continuum) & (self.wave<lineCenter+window_continuum)
        line_selection = (self.wave>lineCenter-5) & (self.wave<lineCenter+5)
        region = continuum_selection ^ line_selection
        continuum = np.nanmedian(f_nu[region])
        lineAmp = ew * continuum / (sigma * np.sqrt(2*np.pi))
        return lineAmp

    def halpha_model(self,f_nu,ew_ha,logzsol=0,ebv=0,alpha=0,redshift=2,csi=0.5,OIIIrat=3,o3o2=0.35,OIIrat=1.4,added_lines=True):
        fwhm = 10 ## AA (see Faisst et al. 2016, sect. 3.2.2)
        fwhm_weak = 5  ## AA (see Faisst et al. 2016, sect. 3.2.2)
        z_pivot = 2

        sigma = fwhm / (2*np.sqrt(2*np.log(2)))

        OII_lines = [3726,3729]
        OIII_lines = [4960,5007]
        Hbeta = 4861.4
        Halpha = 6564.5

        ew_haz = ew_ha * ((1+redshift)/(1+z_pivot))**(alpha)

        ## total_flux = A*np.sqrt(2*np.pi)*sigma
        Halpha_amp = self._get_amplitude_from_ew(f_nu,Halpha,ew_haz,sigma)
        self.add_emission_line(Halpha,amp=Halpha_amp,fwhm=fwhm)


        kalpha = klambda_salim_highz(Halpha/1e4)
        kbeta = klambda_salim_highz(Hbeta/1e4)
        Hbeta_amp = 10**(-0.4*ebv*(kbeta-kalpha)) * Halpha_amp / 2.86
        self.add_emission_line(Hbeta,amp=Hbeta_amp,fwhm=fwhm)


        OIII4960_amp = csi / (OIIIrat+1) * Halpha_amp
        self.add_emission_line(OIII_lines[0],amp=OIII4960_amp,fwhm=fwhm)
        OIII5007_amp = OIIIrat * OIII4960_amp
        self.add_emission_line(OIII_lines[1],amp=OIII5007_amp,fwhm=fwhm)

        OII3726_amp = (OIII5007_amp) / o3o2 / (OIIrat+1)
        self.add_emission_line(OII_lines[0],amp=OII3726_amp,fwhm=fwhm)
        OII3729_amp = OII3726_amp * OIIrat
        self.add_emission_line(OII_lines[1],amp=OII3729_amp,fwhm=fwhm)


        if added_lines is True:
            met_weights = self._compute_weights(logzsol)
            fratio =   met_weights[0] * self.line_data["z0004"]\
                     + met_weights[1] * self.line_data["z004"]\
                     + met_weights[2] * self.line_data["z008"]
            for i in range(len(self.line_data)):
                amp = Hbeta_amp * fratio[i]
                self.add_emission_line(self.line_data["lambda_c"][i],amp=amp,fwhm=fwhm_weak)
        return
