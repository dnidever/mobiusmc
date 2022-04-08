#!/usr/bin/env python

"""SAMPLER.PY - Sampler for periodic signals such as variable stars

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220320'  # yyyymmdd

import time
import numpy as np
from dlnpyutils import utils as dln
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import copy
import emcee
import corner


class VariableSampler:
    """
    Class for doing sampling of variable star lightcurve ddata.

    Parameters
    ----------
    catalog : table
       Catalog of data points, just have mag, err, jd, band
    template : table or function
       Template information as a table with phase and mag columns or
         function/method that takes phase array.
    ampratios : dict, optional
       Amplitude ratios.  Keys should be the unique band names
         and values should be the amplitue ratios.
         If this is not input, then a ratio of 1.0 is used.
    minerror : float, optional
       Minimum error to use.  Default is 0.02.
    
    """

    def __init__(self,catalog,template,ampratios=None,minerror=0.02):

        # Create the sampling for Period (pmin to pmax) and phase offset (0-1)

        self._catalog = catalog
        self.data = Table(catalog).copy()
        for n in self.data.colnames:
            self.data[n].name = n.lower()   # change columns names to lower case
        self.template = Table(template).copy()
        for n in self.template.colnames:
            self.template[n].name = n.lower()   # change columns names to lower case        

        # "filter" not "band" input
        if 'band' not in self.data.colnames and 'filter' in self.data.colnames:
            self.data['band'] = self.data['filter']
        # "mjd" not "jd" input
        if 'jd' not in self.data.colnames and 'mjd' in self.data.colnames:
            self.data['jd'] = self.data['mjd']
        # Check that the catalog and template have the columns that we need
        missingcols = []
        for n in ['mag','err','jd','band']:
            if n not in self.data.colnames:
                missingcols.append(n)
        if len(missingcols)>0:
            raise ValueError('Missing catalog required columns: '+', '.join(missingcols))
        missingcols = []
        for n in ['phase','mag']:
            if n not in self.template.colnames:
                missingcols.append(n)
        if len(missingcols)>0:
            raise ValueError('Missing template required columns: '+', '.join(missingcols))        

        # Make sure the data are sorted by JD
        si = np.argsort(self.data['jd'])
        self.data = self.data[si]
        
        # Add weights to internal catalog
        self.data['wt'] = 1/np.maximum(self.data['err'],minerror)**2
        data = self.data
    
        # Only keep bands with 2+ observations
        uband = np.unique(data['band'])
        badind = np.array([],int)
        for i,b in enumerate(uband):
            ind, = np.where(data['band']==b)
            if len(ind)<2:
                print('band '+str(b)+' only has '+str(len(ind))+' observations.  Not using')
                badind = np.hstack((badind,ind))
        if len(badind)>0:
            data.remove_rows(badind)
        ndata = len(data)
        self.data = data   # the data points that are left
        self.ndata = ndata
        
        print(str(ndata)+' data points')
        print('time baseline = %.2f' % (np.max(data['jd'])-np.min(data['jd'])))
    
        # Get band index
        uband = np.unique(data['band'])
        nband = len(uband)
        bandindex = {}
        for i,b in enumerate(uband):
            ind, = np.where(data['band']==b)
            bandindex[b] = ind            
        self.bands = uband
        self.nbands = nband
        self._bandindex = bandindex
            
        print(str(len(uband))+' bands = ',', '.join(np.char.array(uband).astype(str)))
        
        # No amplitude ratios input
        if ampratios is None:
            ampratios = {}
            for b in uband:
                ampratios[b] = 1.0
        self.ampratios = ampratios
                
        # Pre-calculate some terms that are constant
        totwtdict = {}
        totwtydict = {}
        for b in uband:
            ind = bandindex[b]
            totwtdict[b] = np.sum(data['wt'][ind])
            totwtydict[b] = np.sum(data['wt'][ind] * data['mag'][ind])
        self._totwtdict = totwtdict
        self._totwtydict = totwtydict

    def solve(self,period,offset,amplitude=None):
        """
        Solve for a given period and offset.

        Parameters
        ----------
        period : float or array
          Period as scalar float or array.
        offset : float or array
          Phase offset as scalar float or array.
        amplitude : float or array, optional
          Amplitude.  If this is not input, then the best amplitude
            using linear least squares is derived.

        Returns
        -------
        amplitude : float or array
          The amplitudes.
        constant : float or array
          The best constant offset.
        lnlikelihood : float or array
          The log likelihood.

        Example
        -------

        amp,const,lnlkhood = samp.solve(period,offset)

        """

        nperiod = np.array(period).size
        data = self.data
        template = self.template
        ampratios = self.ampratios
        bandindex = self._bandindex
        totwtdict = self._totwtdict
        totwtydict = self._totwtydict
        
        ndata = len(data)

        # Get phase and template points
        phase = (data['jd'].reshape(-1,1)/period.reshape(1,-1) + offset.reshape(1,-1)) % 1
        if hasattr(template, '__call__'):            
            tmpl = template(phase.ravel())
        else:
            tmpl = np.interp(phase.ravel(),template['phase'],template['mag'])
        tmpl = tmpl.reshape(ndata,nperiod)
            
        # -- Find best fitting values for linear parameters ---
        # Calculate amplitude
        # term1 = Sum of XY
        # term2 = Sum of X * Y / W 
        # term3 = Sum of X^2
        # term4 = Sum of X * X / W
        # amplitude = (term1 - term2)/(term3 - term4)
        if amplitude is None:
            term1,term2,term3,term4 = 0,0,0,0
            if nperiod==1:
                totwtxdict = {}                
                for b in bandindex.keys():
                    ind = bandindex[b]
                    totwtx1 = np.sum(data['wt'][ind] * tmpl[ind]*ampratios[b])
                    totwtxdict[b] = totwtx1
                    totwtx2 = np.sum(data['wt'][ind] * (tmpl[ind]*ampratios[b])**2)
                    totwtxy = np.sum(data['wt'][ind] * tmpl[ind]*ampratios[b] * data['mag'][ind])      
                    term1 += totwtxy
                    term2 += totwtx1 * totwtydict[b] / totwtdict[b]
                    term3 += totwtx2
                    term4 += totwtx1**2 / totwtdict[b]
                amplitude = (term1-term2)/(term3-term4)
            else:
                totwtxdict = {}
                for b in bandindex.keys():
                    ind = bandindex[b]
                    totwtx1 = np.sum(data['wt'][ind].reshape(-1,1) * tmpl[ind,:]*ampratios[b],axis=0)
                    totwtxdict[b] = totwtx1
                    totwtx2 = np.sum(data['wt'][ind].reshape(-1,1) * (tmpl[ind,:]*ampratios[b])**2,axis=0)
                    totwtxy = np.sum(data['wt'][ind].reshape(-1,1) * tmpl[ind,:]*ampratios[b] * data['mag'][ind].reshape(-1,1),axis=0)      
                    term1 += totwtxy
                    term2 += totwtx1 * totwtydict[b] / totwtdict[b]
                    term3 += totwtx2
                    term4 += totwtx1**2 / totwtdict[b]
                amplitude = (term1-term2)/(term3-term4)
        else:
            if nperiod==1:
                for b in bandindex.keys():
                    ind = bandindex[b]
                    totwtx1 = np.sum(data['wt'][ind] * tmpl[ind]*ampratios[b])
                    totwtxdict[b] = totwtx1
            else:
                totwtxdict = {}
                for b in bandindex.keys():
                    ind = bandindex[b]
                    totwtx1 = np.sum(data['wt'][ind].reshape(-1,1) * tmpl[ind,:]*ampratios[b],axis=0)
                    totwtxdict[b] = totwtx1
                
                
        # Calculate best mean magnitudes
        # mean mag = (Y - amplitude * X)/W
        meanmag = {}
        for b in bandindex.keys():
            meanmag1 = (totwtydict[b] - amplitude * totwtxdict[b])/totwtdict[b]
            meanmag[b] = meanmag1
            
        # Calculate likelihood/chisq
        if nperiod==1:
            model = np.zeros(ndata,float)
            resid = np.zeros(ndata,float)
            wtresid = np.zeros(ndata,float)        
            for b in bandindex.keys():
                ind = bandindex[b]          
                model1 = tmpl[ind]*ampratios[b]*amplitude+meanmag[b]
                model[ind] = model1
                resid[ind] = data['mag'][ind]-model1
                wtresid[ind] = resid[ind]**2 * data['wt'][ind]
            lnlkhood = -0.5*np.sum(wtresid + np.log(2*np.pi*data['err']**2))
        else:
            model = np.zeros((ndata,nperiod),float)
            resid = np.zeros((ndata,nperiod),float)
            wtresid = np.zeros((ndata,nperiod),float)        
            for b in bandindex.keys():
                ind = bandindex[b]
                model1 = tmpl[ind,:]*ampratios[b]*amplitude.reshape(1,-1)+meanmag[b].reshape(1,-1)
                model[ind,:] = model1
                resid[ind,:] = data['mag'][ind].reshape(-1,1)-model1
                wtresid[ind,:] = resid[ind,:]**2 * data['wt'][ind].reshape(-1,1)
            lnlikelihood = -0.5*np.sum(wtresid,axis=0)
            lnlikelihood += -0.5*np.sum(np.log(2*np.pi*data['err']**2))

        return amplitude,meanmag,lnlikelihood
    
              
    def copy(self):
        """ Make a copy."""
        return copy.deepcopy(self)
        
    def run(self,pmin=0.1,pmax=None,offsetrange=None,minsample=128,npoints=200000,
            unirefine=True,keepnegamp=False,verbose=True):
        """
        Run the sampler.

        Parameters
        ----------
        pmin : float, optional
           Minimum period to search in days.  Default is 0.1 days.
        pmax : float, optional
           Maximum period to search in days.  Default is 2 x time baseline.
        offsetrange : list, optional
           Two-element range of phase offset values to explore.  Default is [0,1].
        minsample : int, optional
           Mininum number of samples to return.  Default is 128.
        npoints : int, optional
           Number of points to use per loop.  Default is 200,000.
        unirefine : boolean, optional
           If a unimodal posterior distribution function, do a finer search
             around the unimodal region.  Default is True.
        keepnegamp : boolean, optional
           Keep negative amplitudes.  Default is False.
        verbose : boolean, optional
           Print useful information to the screen.  Default is True.

        Returns
        -------
        samples : astropy table
           The Monte Carlo samples that passed rejection sampling.
             period, offset, amplitude, lnlikelihood, lnprob, meanmaxBAND.
        trials: astropy table
           All of the trials period and phase offset positions tried.
             period, offset, amplitude, lnlikelihood, lnprob, meanmaxBAND.
        best : dictionary
           Dictionary of best values (in ln probability) across all of
             the trials: period, offset, amplitude, meanmag, lnprob.

        Example
        -------

        samples,trials,best = vs.run()

        """

        data = self.data
        ndata = self.ndata
        template = self.template
        uband = self.bands
        nband = self.nbands
        ampratios = self.ampratios        
        bandindex = self._bandindex
        totwtdict = self._totwtdict
        totwtydict = self._totwtydict

        self.bestperiod = None
        self.bestoffset = None
        self.bestamplitude = None
        self.bestmeanmag = None
        self.bestlnprob = None        
        self.samples = None
        self.trials = None
        
        # Period range
        if pmax is None:
            pmax = (np.max(data['jd'])-np.min(data['jd']))*2
        lgminp = np.log10(pmin)
        lgmaxp = np.log10(pmax)

        if verbose:
            print('Pmin = %.3f' % pmin)
            print('Pmax = %.3f' % pmax)    
        self._pmin = pmin
        self._pmax = pmax

        # Phase offset range
        if offsetrange is not None:
            offsetmin = offsetrange[0]
            offsetmax = offsetrange[1]
        else:
            offsetmin = 0
            offsetmax = 1
        if offsetmin<0 or offsetmax>1:
            raise ValueError('Phase offset range must be within 0 to 1')
        if verbose:
            print('Phase offset min = %.3f' % offsetmin)
            print('Phase offset max = %.3f' % offsetmax)            
        
        # Loop until we have enough samples
        nsamples = 0
        samplelist = []
        count = 0
        dtt = [('period',float),('offset',float),('amplitude',float),('lnlikelihood',float),('lnprob',float)]
        for b in uband:
            dtt += [('mag'+str(b),float)]
        trials = None
        while (nsamples<minsample):
    
            # Uniformly sample from log(pmin) to log(pmax)
            period = np.random.rand(npoints)*(lgmaxp-lgminp)+lgminp    
            period = 10**period
            # Uniformly sample from offsetmin to offsetmax
            offset = np.random.rand(npoints)*(offsetmax-offsetmin)+offsetmin

            # Solve for amplitude, constant and lnlikelihood
            amplitude,meanmag,lnlikelihood = self.solve(period,offset)

            # Calculate ln probability = ln prior + ln likelihood
            # use flat prior, divide by area
            lnprior = np.ones(npoints,float) + np.log(1/((lgmaxp-lgminp)*(offsetmax-offsetmin)))
            lnprob = lnprior + lnlikelihood

            # Save the information
            trials1 = np.zeros(npoints,dtype=dtt)
            trials1['period'] = period
            trials1['offset'] = offset
            trials1['amplitude'] = amplitude
            for k in meanmag.keys():
                trials1['mag'+str(k)] = meanmag[k]
            trials1['lnlikelihood'] = lnlikelihood
            trials1['lnprob'] = lnprob        
            if trials is None:
                trials = trials1
            else:
                trials = np.hstack((trials,trials1))
        
            # REJECT NEGATIVE AMPLITUDES??
        
            # Rejection sampling
            draw = np.random.rand(npoints)
            if keepnegamp is False:
                ind, = np.where((draw < np.exp(lnprob))  & (amplitude > 0))
            else:
                ind, = np.where(draw < np.exp(lnprob))                
            if len(ind)>0:
                for i in ind:
                    samp = {'period':period[i],'offset':offset[i],'amplitude':amplitude[i]}
                    for k in meanmag.keys():
                        samp[k] = meanmag[k][i]
                    samp['lnlikelihood'] = lnlikelihood[i]
                    samp['lnprob'] = lnprob[i]
                    samplelist.append(samp)
                nsamples += len(ind)

            if verbose:
                print(count+1,nsamples)
            count += 1
        
        # Convert sample list to table
        dt = [('period',float),('offset',float),('amplitude',float)]
        for k in meanmag.keys():
            dt += [('mag'+str(k),float)]
        dt += [('lnlikelihood',float),('lnprob',float)]
        samples = np.zeros(len(samplelist),dtype=dt)
        for i,samp in enumerate(samplelist):
            samples['period'][i] = samp['period']
            samples['offset'][i] = samp['offset']
            samples['amplitude'][i] = samp['amplitude']
            samples['lnlikelihood'][i] = samp['lnlikelihood']
            samples['lnprob'][i] = samp['lnprob']
            for k in meanmag.keys():
                samples['mag'+str(k)][i] = samp[k]

        # Convert to astropy tables
        samples = Table(samples)
        trials = Table(trials)
        self.samples = samples
        self.trials = trials

        if keepnegamp is False:
            posind, = np.where(trials['amplitude']>0)
            best1 = np.argmax(trials['lnprob'][posind])
            best = posind[best1]
        else:
            best = np.argmax(trials['lnprob'])                
        bestperiod = trials['period'][best]
        bestoffset = trials['offset'][best]
        bestlnprob = trials['lnprob'][best]
        bestamplitude = trials['amplitude'][best]
        bestmeanmag = {}
        for b in uband:
            bestmeanmag[b] = trials['mag'+str(b)][best]
        if verbose:
            print('Best period = %.4f' % bestperiod)
            print('Best offset = %.4f' % bestoffset)
            print('Best amplitude = %.4f' % bestamplitude)
            print('Best lnprob = %.4f' % bestlnprob)
        self.bestperiod = bestperiod
        self.bestoffset = bestoffset
        self.bestamplitude = bestamplitude
        self.bestmeanmag = bestmeanmag
        self.bestlnprob = bestlnprob
        
        ntrials = npoints*count
        if verbose:
            print('ntrials = ',ntrials)
    
        
        # If unimodal, run higher sampling
        medperiod = np.median(samples['period'])
        delta = (4*medperiod**2)/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        deltap = medperiod**2/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        rmsperiod = np.sqrt(np.mean((samples['period']-medperiod)**2))
        unimodal = False
        if rmsperiod < delta and unirefine:
            print('Unimodal PDF, finer sampling')
            unimodal = True
            # Fine sampling around the maximum
            rmsperiod = dln.mad(samples['period'].data-medperiod,zero=True)
            pmin2 = np.maximum(medperiod - 3*rmsperiod,pmin)
            pmax2 = medperiod + 3*rmsperiod
            medoffset = np.median(samples['offset'])
            rmsoffset = dln.mad(samples['offset'].data-medoffset,zero=True)
            offsetmin2 = np.maximum(medoffset-3*rmsoffset,offsetmin)
            offsetmax2 = medoffset+3*rmsoffset
            medamplitude = np.median(samples['amplitude'])
            rmsamplitude = dln.mad(samples['amplitude'].data-medamplitude,zero=True)
            ampmin2 = np.maximum(medamplitude-3*rmsamplitude,0)
            ampmax2 = medamplitude+3*rmsamplitude
            # Uniformly sample from min to max            
            period2 = np.random.rand(npoints)*(pmax2-pmin2)+pmin2
            offset2 = np.random.rand(npoints)*(offsetmax2-offsetmin2)+offsetmin2
            amplitude2 = np.random.rand(npoints)*(ampmax2-ampmin2)+ampmin2
            # Calculate amplitude, constant and lnlikelihood
            amplitude3,meanmag2,lnlikelihood2 = self.solve(period2,offset2,amplitude2)
            # Calculate ln probability = ln prior + ln likelihood
            # use flat prior, divide by area
            lnprior2 = np.ones(npoints,float) + np.log(1/((pmax2-pmin2)*(offsetmax2-offsetmin2)*(ampmax2-ampmin2)))
            lnprob2 = lnprior2 + lnlikelihood2

            # Save trial information
            trials0 = trials
            del trials
            trials = np.zeros(npoints,dtype=dtt)
            trials['period'] = period2
            trials['offset'] = offset2
            trials['amplitude'] = amplitude2
            for k in meanmag.keys():
                trials['mag'+str(k)] = meanmag2[k]
            trials['lnlikelihood'] = lnlikelihood2
            trials['lnprob'] = lnprob2
            
            # Rejection sampling
            draw = np.random.rand(npoints)
            if keepnegamp is False:
                ind, = np.where((draw < np.exp(lnprob))  & (amplitude > 0))
            else:
                ind, = np.where(draw < np.exp(lnprob))                
            if len(ind)>0:
                # Convert sample list to table
                dt = [('period',float),('offset',float),('amplitude',float)]
                for k in meanmag.keys():
                    dt += [('mag'+str(k),float)]
                dt += [('lnlikelihood',float),('lnprob',float)]
                samples = np.zeros(len(ind),dtype=dt)
                samples['period'] = period2[ind]
                samples['offset'] = offset2[ind]
                samples['amplitude'] = amplitude2[ind]
                samples['lnlikelihood'] = lnlikelihood2[ind]
                samples['lnprob'] = lnprob2[ind]
                for k in meanmag.keys():
                    samples['mag'+str(k)] = meanmag[k][ind]

            samples = Table(samples)
            trials = Table(trials)
            self.samples = samples
            self.trials = trials

            # Get best values
            best = np.argmax(trials['lnprob'])
            bestperiod = trials['period'][best]
            bestoffset = trials['offset'][best]
            bestlnprob = trials['lnprob'][best]
            bestamplitude = trials['amplitude'][best]
            bestmeanmag = {}
            for b in uband:
                bestmeanmag[b] = trials['mag'+str(b)][best]
            if verbose:
                print('Best period = %.4f' % bestperiod)
                print('Best offset = %.4f' % bestoffset)
                print('Best amplitude = %.4f' % bestamplitude)
                for b in uband:
                    print('Best meanmag %s = %.4f' %(str(b),bestmeanmag[b]))
                print('Best lnprob = %.4f' % bestlnprob)                
            self.bestperiod = bestperiod
            self.bestoffset = bestoffset
            self.bestamplitude = bestamplitude
            self.bestmeanmag = bestmeanmag
            self.bestlnprob = bestlnprob

        self.unimodal = unimodal

        # Construct best dictionary
        best = {'period':bestperiod,'phase':bestoffset,'amplitude':bestamplitude,
                'meanmag':bestmeanmag,'lnprob':bestlnprob}

        return samples, trials, best
        

    def plots(self,plotbase='sampler',bins=(200,200)):
        """ Make the plots."""

        data = self.data
        ndata = self.ndata
        template = self.template
        uband = self.bands
        nband = self.nbands
        bandindex = self._bandindex
        ampratios = self.ampratios
        bestperiod = self.bestperiod
        bestoffset = self.bestoffset
        bestamplitude = self.bestamplitude
        bestmeanmag = self.bestmeanmag
        bestlnprob = self.bestlnprob
        samples = self.samples
        trials = self.trials
        
        
        # Make plots
        matplotlib.use('Agg')
        fig,ax = plt.subplots(2,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        xr = [np.min(np.log10(trials['period'])),np.max(np.log10(trials['period']))]
        # 2D density map
        im,b,c,d = stats.binned_statistic_2d(trials['offset'],np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=bins)
        z1 = ax[0].imshow(im,aspect='auto',origin='lower',extent=(c[0],c[-1],b[0],b[-1]))
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        ax[0].set_xlim(xr)
        plt.colorbar(z1,ax=ax[0],label='Mean ln(Prob)')
        # Period histogram
        hist,a,b = stats.binned_statistic(np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=1000)
        ax[1].plot(a[0:-1],hist)
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Mean ln(Prob)')
        ax[1].set_xlim(xr)        
        fig.savefig(plotbase+'_trials.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_trials.png')

        # Plot offset vs. period color-coded by lnprob
        # plot amplitude vs. period color-coded by lnprob
        fig,ax = plt.subplots(3,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        xr = [np.min(samples['period']),np.max(samples['period'])]        
        # Plot offset vs. period color-coded by lnprob
        z1 = ax[0].scatter(np.log10(samples['period']),samples['offset'],c=samples['lnprob'])
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        ax[0].set_xlim(xr)
        plt.colorbar(z1,ax=ax[0],label='ln(Prob)')
        # Plot amplitude vs. period color-coded by lnprob
        z2 = ax[1].scatter(np.log10(samples['period']),samples['amplitude'],c=samples['lnprob'])
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_xlim(xr)        
        plt.colorbar(z2,ax=ax[1],label='ln(Prob)')
        # Sum of lnprob
        hist2,a2,b2 = stats.binned_statistic(np.log10(samples['period']),samples['lnprob'],statistic='sum',bins=50)
        ax[2].plot(a2[0:-1],hist2)
        ax[2].set_xlabel('log(Period)')
        ax[2].set_ylabel('Sum ln(Prob)')
        ax[2].set_xlim(xr)        
        fig.savefig(plotbase+'_samples.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_samples.png')
        
        # Plot best-fit model
        # one panel per band, mag vs. phase
        fig,ax = plt.subplots(nband,1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        phase = (data['jd']/bestperiod + bestoffset) % 1
        if hasattr(template, '__call__'):            
            tmpl = template(phase.ravel())
        else:
            tmpl = np.interp(phase,template['phase'],template['mag'])
        for i,b in enumerate(uband):
            ind = bandindex[b]
            tphase = (np.linspace(0,1,100)+bestoffset) % 1
            si = np.argsort(tphase)
            tphase = tphase[si]
            tmag = np.interp(tphase,template['phase'],template['mag'])
            model = tmag*ampratios[b]*bestamplitude+bestmeanmag[b]
            dd = np.hstack((data['mag'][ind],model))
            yr = [np.max(dd)+0.05*dln.valrange(dd),np.min(dd)-0.30*dln.valrange(dd)]
            ax[i].plot(tphase,model,c='blue',zorder=1)        
            ax[i].errorbar(phase[ind],data['mag'][ind],yerr=data['err'][ind],c='gray',fmt='none',zorder=2)
            ax[i].scatter(phase[ind],data['mag'][ind],c='black',zorder=3)
            txt = 'Band '+str(b)
            if ampratios is not None:
                txt += '  Amp Ratio=%.3f' % ampratios[b]
            ax[i].annotate(txt,xy=(0.02,yr[1]+0.10*dln.valrange(dd)),ha='left')
            ax[i].annotate('Amplitude=%.3f' % (bestamplitude*ampratios[b]),xy=(0.02,yr[1]+0.20*dln.valrange(dd)),ha='left')
            ax[i].annotate('Mean Mag=%.3f' % bestmeanmag[b],xy=(0.02,yr[1]+0.30*dln.valrange(dd)),ha='left')                    
            ax[i].set_xlabel('Phase')
            ax[i].set_ylabel('Magnitude')
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(yr)
            if i==0:
                ax[i].set_title('Period=%.3f  Offset=%.3f  Amplitude=%.3f  ln(Prob)=%.3f' %
                                (bestperiod,bestoffset,bestamplitude,bestlnprob))
        fig.savefig(plotbase+'_best.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_best.png')    

#--------------------------------------------------------------------------------------------------------------------


class LinearModelSampler:
    """
    Class to perform sampling of periodic linear model (y=a*model(phase)+b)

    Parameters
    ----------
    data : table
       Tuple with (x,y,yerr) or table with columns x, y and yerr.
    model : function or table
       Model function or template with x and y columns.
    minerror : float, optional
       Minimum error to use.  Default is 0.02.
    
    """

    def __init__(self,data,model,minerror=0.02):

        # Create the sampling for Period (pmin to pmax) and phase offset (0-1)

        if type(data) is tuple:
            temp = np.zeros(len(data[0]),dtype=np.dtype([('x',float),('y',float),('yerr',float)]))
            temp['x'] = data[0]
            temp['y'] = data[1]
            temp['yerr'] = data[2]            
            self.data = Table(temp)
        else:
            self.data = Table(data).copy()
        for n in self.data.colnames:
            self.data[n].name = n.lower()   # change columns names to lower case
        self.model = model
        
        # Add weights to internal catalog
        self.data['wt'] = 1/np.maximum(self.data['yerr'],minerror)**2
        data = self.data

        # Make sure the data are sorted by x
        si = np.argsort(self.data['x'])
        self.data = self.data[si]
        
        print(str(ndata)+' data points')
        print('time baseline = %.2f' % (np.max(data['jd'])-np.min(data['jd'])))
                
        # Pre-calculate some terms that are constant
        totwt = np.sum(data['wt'])
        totwty = np.sum(data['wt']*data['y'])
        self._totwt = totwt
        self._totwty = totwty

    def solve(self,period,offset,amplitude=None):
        """ 
        Solve for a given period and offset.

        Parameters
        ----------
        period : float or array
          Period as scalar float or array.
        offset : float or array
          Phase offset as scalar float or array.
        amplitude : float or array, optional
          Amplitude.  If this is not input, then the best amplitude
            using linear least squares is derived.

        Returns
        -------
        amplitude : float or array
          The amplitudes.
        constant : float or array
          The best constant offset.
        lnlikelihood : float or array
          The log likelihood.

        Example
        -------

        amp,const,lnlkhood = samp.solve(period,offset)

        """

        data = self.data
        ndata = len(data)
        nperiod = np.array(period).size
        
        # Calculate phase for each data point
        if nperiod==1:
            phase = (data['x']/period + offset) % 1
        else:
            phase = (data['x'].reshape(-1,1)/period.reshape(1,-1) + offset.reshape(1,-1)) % 1
            
        # Calculate template values for this set of period and phase
        if hasattr(model, '__call__'):            
            tmpl = model(phase)
        else:
            tmpl = np.interp(phase,model['x'],model['y'])
        if nperiod>1:
            tmpl = tmpl.reshape(ndata,nperiod)
            
        # -- Find best fitting values for linear parameters ---
        # Calculate amplitude
        # term1 = Sum of XY
        # term2 = Sum of X * Y / W 
        # term3 = Sum of X^2
        # term4 = Sum of X * X / W
        # amplitude = (term1 - term2)/(term3 - term4)
        if amplitude is None:
            if nperiod==1:
                totwtx1 = np.sum(data['wt'] * tmpl)
                totwtx = totwtx1
                totwtx2 = np.sum(data['wt'] * tmpl**2)
                totwtxy = np.sum(data['wt'] * tmpl*data['y']) 
                term1 = totwtxy
                term2 = totwtx * totwty / totwt
                term3 = totwtx2
                term4 = totwtx**2 / totwt
                amplitude = (term1-term2)/(term3-term4)
            else:
                totwtx = np.sum(data['wt'].reshape(-1,1) * tmpl,axis=0)
                totwtx2 = np.sum(data['wt'].reshape(-1,1) * tmpl**2,axis=0)
                totwtxy = np.sum(data['wt'].reshape(-1,1) * tmpl * data['y'].reshape(-1,1),axis=0)      
                term1 += totwtxy
                term2 += totwtx * totwty / totwt
                term3 += totwtx2
                term4 += totwtx**2 / totwt
                amplitude = (term1-term2)/(term3-term4)

        
        # Calculate best constant value
        # constant = (Y - amplitude * X)/W
        constant = (totwty-amplitude*totwtx)/totwt

        # Calculate likelihood
        if nperiod==1:
            model1 = tmpl*amplitude+constant
            resid = data['y']-model1
            wtresid = resid**2 * data['wt']
            lnlikelihood = -0.5*np.sum(wtresid + np.log(2*np.pi*data['yerr']**2))
        else:
            model1 = tmpl*amplitude.reshape(1,-1)+constant.reshape(1,-1)
            resid = data['y'].reshape(-1,1)-model1
            wtresid = resid**2 * data['wt'].reshape(-1,1)
            lnlikelihood = -0.5*np.sum(wtresid,axis=0)
            lnlikelihood += -0.5*np.sum(np.log(2*np.pi*data['yerr']**2))

        return amplitude,constant,lnlikelihood

    
    def copy(self):
        """ Make a copy."""
        return copy.deepcopy(self)
        
    def run(self,pmin=0.1,pmax=None,offsetrange=None,minsample=128,npoints=200000,
            unirefine=True,keepnegamp=False,verbose=True):
        """
        Run the sampler.

        Parameters
        ----------
        pmin : float, optional
           Minimum period to search in days.  Default is 0.1 days.
        pmax : float, optional
           Maximum period to search in days.  Default is 2 x time baseline.
        offsetrange : list, optional
           Two-element range of phase offset values to explore.  Default is [0,1].
        minsample : int, optional
           Mininum number of samples to return.  Default is 128.
        npoints : int, optional
           Number of points to use per loop.  Default is 200,000.
        unirefine : boolean, optional
           If a unimodal posterior distribution function, do a finer search
             around the unimodal region.  Default is True.
        keepnegamp : boolean, optional
           Keep negative amplitudes.  Default is False.
        verbose : boolean, optional
           Print useful information to the screen.  Default is True.

        Returns
        -------
        samples : astropy table
           The Monte Carlo samples that passed rejection sampling.
             period, offset, amplitude, lnlikelihood, lnprob, meanmaxBAND.
        trials: astropy table
           All of the trials period and phase offset positions tried.
             period, offset, amplitude, lnlikelihood, lnprob, meanmaxBAND.
        best : dictionary
           Dictionary of best values (in ln probability) across all of
             the trials: period, offset, amplitude, meanmag, lnprob.

        Example
        -------

        samples,trials,best = vs.run()

        """

        data = self.data
        ndata = self.ndata
        model = self.model
        totwt = self._totwt
        totwty = self._totwty

        self.bestperiod = None
        self.bestoffset = None
        self.bestamplitude = None
        self.bestconstant = None
        self.bestlnprob = None        
        self.samples = None
        self.trials = None
        
        # Period range
        if pmax is None:
            pmax = (np.max(data['x'])-np.min(data['x']))*2
        lgminp = np.log10(pmin)
        lgmaxp = np.log10(pmax)

        if verbose:
            print('Pmin = %.3f' % pmin)
            print('Pmax = %.3f' % pmax)    
        self.pmin = pmin
        self.pmax = pmax

        # Phase offset range
        if offsetrange is not None:
            offsetmin = offsetrange[0]
            offsetmax = offsetrange[1]
        else:
            offsetmin = 0
            offsetmax = 1
        if offsetmin<0 or offsetmax>1:
            raise ValueError('Phase offset range must be within 0 to 1')
        if verbose:
            print('Phase offset min = %.3f' % offsetmin)
            print('Phase offset max = %.3f' % offsetmax)            
        
        # Loop until we have enough samples
        nsamples = 0
        samplelist = []
        count = 0
        dtt = [('period',float),('offset',float),('amplitude',float),('constant',float),('lnlikelihood',float),('lnprob',float)]
        trials = None
        while (nsamples<minsample):
    
            # Uniformly sample from log(pmin) to log(pmax)
            period = np.random.rand(npoints)*(lgmaxp-lgminp)+lgminp    
            period = 10**period
            # Uniformly sample from offsetmin to offsetmax
            offset = np.random.rand(npoints)*(offsetmax-offsetmin)+offsetmin

            # Solve for amplitude, constant and lnlikelihood
            amplitude,constant,lnlikelihood = self.solve(period,offset)

            # Calculate ln probability = ln prior + ln likelihood
            # use flat prior, divide by area
            lnprior = np.ones(npoints,float) + np.log(1/(1.0*(lgmaxp-lgminp)))
            lnprob = lnprior + lnlikelihood

            # Save the information
            trials1 = np.zeros(npoints,dtype=dtt)
            trials1['period'] = period
            trials1['offset'] = offset
            trials1['amplitude'] = amplitude
            trials1['constant'] = constant
            trials1['lnlikelihood'] = lnlikelihood
            trials1['lnprob'] = lnprob        
            if trials is None:
                trials = trials1
            else:
                trials = np.hstack((trials,trials1))
                
            # Rejection sampling
            draw = np.random.rand(npoints)
            if keepnegamp is False:
                ind, = np.where((draw < np.exp(lnprob))  & (amplitude > 0))
            else:
                ind, = np.where(draw < np.exp(lnprob))                
            if len(ind)>0:
                for i in ind:
                    samp = {'period':period[i],'offset':offset[i],'amplitude':amplitude[i],'constant':constant[i]}
                    samp['lnlikelihood'] = lnlikelihood[i]
                    samp['lnprob'] = lnprob[i]
                    samplelist.append(samp)
                nsamples += len(ind)

            if verbose:
                print(count+1,nsamples)
            count += 1
        
        # Convert sample list to table
        dt = [('period',float),('offset',float),('amplitude',float),('constant',float)]
        dt += [('lnlikelihood',float),('lnprob',float)]
        samples = np.zeros(len(samplelist),dtype=dt)
        for i,samp in enumerate(samplelist):
            samples['period'][i] = samp['period']
            samples['offset'][i] = samp['offset']
            samples['amplitude'][i] = samp['amplitude']
            samples['constant'][i] = samp['constant']
            samples['lnlikelihood'][i] = samp['lnlikelihood']
            samples['lnprob'][i] = samp['lnprob']

        # Convert to astropy tables
        samples = Table(samples)
        trials = Table(trials)
        self.samples = samples
        self.trials = trials

        if keepnegamp is False:
            posind, = np.where(trials['amplitude']>0)
            best1 = np.argmax(trials['lnprob'][posind])
            best = posind[best1]
        else:
            best = np.argmax(trials['lnprob'])                
        bestperiod = trials['period'][best]
        bestoffset = trials['offset'][best]
        bestlnprob = trials['lnprob'][best]
        bestamplitude = trials['amplitude'][best]
        bestconstant = trials['constant'][best]
        if verbose:
            print('Best period = %.4f' % bestperiod)
            print('Best offset = %.4f' % bestoffset)
            print('Best amplitude = %.4f' % bestamplitude)
            print('Best constant = %.4f' % bestconstant)            
            print('Best lnprob = %.4f' % bestlnprob)
        self.bestperiod = bestperiod
        self.bestoffset = bestoffset
        self.bestamplitude = bestamplitude
        self.bestconstant = bestconstant        
        self.bestlnprob = bestlnprob
        
        ntrials = npoints*count
        if verbose:
            print('ntrials = ',ntrials)
    
        
        # If unimodal, run emcee
        medperiod = np.median(samples['period'])
        delta = (4*medperiod**2)/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        deltap = medperiod**2/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        rmsperiod = np.sqrt(np.mean((samples['period']-medperiod)**2))
        unimodal = False
        if rmsperiod < delta and unirefine:
            print('Unimodal PDF, finer sampling')
            unimodal = True
            # Fine sampling around the maximum
            rmsperiod = dln.mad(samples['period'].data-medperiod,zero=True)
            pmin2 = np.maximum(medperiod - 3*rmsperiod,pmin)
            pmax2 = medperiod + 3*rmsperiod
            medoffset = np.median(samples['offset'])
            rmsoffset = dln.mad(samples['offset'].data-medoffset,zero=True)
            offsetmin2 = np.maximum(medoffset-3*rmsoffset,offsetmin)
            offsetmax2 = medoffset+3*rmsoffset
            medamplitude = np.median(samples['amplitude'])
            rmsamplitude = dln.mad(samples['amplitude'].data-medamplitude,zero=True)
            ampmin2 = np.maximum(medamplitude-3*rmsamplitude,0)
            ampmax2 = medamplitude+3*rmsamplitude
            # Uniformly sample from min to max            
            period2 = np.random.rand(npoints)*(pmax2-pmin2)+pmin2
            offset2 = np.random.rand(npoints)*(offsetmax2-offsetmin2)+offsetmin2
            amplitude2 = np.random.rand(npoints)*(ampmax2-ampmin2)+ampmin2
            # Calculate amplitude, constant and lnlikelihood
            amplitude3,constant2,lnlikelihood2 = self.solve(period2,offset2,amplitude2)
            # Calculate ln probability = ln prior + ln likelihood
            # use flat prior, divide by area
            lnprior2 = np.ones(npoints,float) + np.log(1/((pmax2-pmin2)*(offsetmax2-offsetmin2)*(ampmax2-ampmin2)))
            lnprob2 = lnprior2 + lnlikelihood2

            # Save trial information
            trials0 = trials
            del trials
            trials = np.zeros(npoints,dtype=dtt)
            trials['period'] = period2
            trials['offset'] = offset2
            trials['amplitude'] = amplitude2
            trials['constant'] = constant2
            trials['lnlikelihood'] = lnlikelihood2
            trials['lnprob'] = lnprob2
            
            # Rejection sampling
            draw = np.random.rand(npoints)
            if keepnegamp is False:
                ind, = np.where((draw < np.exp(lnprob))  & (amplitude > 0))
            else:
                ind, = np.where(draw < np.exp(lnprob))                
            if len(ind)>0:
                # Creat table
                dt = [('period',float),('offset',float),('amplitude',float),('constant',float),
                      ('lnlikelihood',float),('lnprob',float)]
                samples = np.zeros(len(ind),dtype=dt)
                samples['period'] = period2[ind]
                samples['offset'] = offset2[ind]
                samples['amplitude'] = amplitude2[ind]
                samples['constant'] = constant2[ind]
                samples['lnlikelihood'] = lnlikelihood2[ind]
                samples['lnprob'] = lnprob2[ind] 

            samples = Table(samples)
            trials = Table(trials)
            self.samples = samples
            self.trials = trials

            # Get best values
            best = np.argmax(trials['lnprob'])
            bestperiod = trials['period'][best]
            bestoffset = trials['offset'][best]
            bestamplitude = trials['amplitude'][best]
            bestconstant = trials['constant'][best]
            bestlnprob = trials['lnprob'][best]
            if verbose:
                print('Best period = %.4f' % bestperiod)
                print('Best offset = %.4f' % bestoffset)
                print('Best amplitude = %.4f' % bestamplitude)
                print('Best constant = %.4f' % bestconstant) 
                print('Best lnprob = %.4f' % bestlnprob)                
            self.bestperiod = bestperiod
            self.bestoffset = bestoffset
            self.bestamplitude = bestamplitude
            self.bestconstant = bestconstant
            self.bestlnprob = bestlnprob

        self.unimodal = unimodal

        # Construct best dictionary
        best = {'period':bestperiod,'phase':bestoffset,'amplitude':bestamplitude,
                'constant':bestconstant,'lnprob':bestlnprob}

        return samples, trials, best


    def plots(self,plotbase='sampler',bins=(200,200)):
        """ Make the plots."""

        data = self.data
        ndata = self.ndata
        model = self.model
        bestperiod = self.bestperiod
        bestoffset = self.bestoffset
        bestamplitude = self.bestamplitude
        bestconstant = self.bestconstant
        bestlnprob = self.bestlnprob
        samples = self.samples
        trials = self.trials
        
        
        # Make plots
        matplotlib.use('Agg')
        fig,ax = plt.subplots(2,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        xr = [np.min(np.log10(trials['period'])),np.max(np.log10(trials['period']))]
        # 2D density map
        im,b,c,d = stats.binned_statistic_2d(trials['offset'],np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=bins)
        z1 = ax[0].imshow(im,aspect='auto',origin='lower',extent=(c[0],c[-1],b[0],b[-1]))
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        ax[0].set_xlim(xr)
        plt.colorbar(z1,ax=ax[0],label='Mean ln(Prob)')
        # Period histogram
        hist,a,b = stats.binned_statistic(np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=1000)
        ax[1].plot(a[0:-1],hist)
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Mean ln(Prob)')
        ax[1].set_xlim(xr)        
        fig.savefig(plotbase+'_trials.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_trials.png')

        # Plot offset vs. period color-coded by lnprob
        # plot amplitude vs. period color-coded by lnprob
        fig,ax = plt.subplots(3,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        xr = [np.min(samples['period']),np.max(samples['period'])]
        # Plot offset vs. period color-coded by lnprob
        z1 = ax[0].scatter(np.log10(samples['period']),samples['offset'],c=samples['lnprob'])
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        ax[0].set_xlim(xr)
        plt.colorbar(z1,ax=ax[0],label='ln(Prob)')
        # Plot amplitude vs. period color-coded by lnprob
        z2 = ax[1].scatter(np.log10(samples['period']),samples['amplitude'],c=samples['lnprob'])
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_xlim(xr)        
        plt.colorbar(z2,ax=ax[1],label='ln(Prob)')
        # Sum of lnprob
        hist2,a2,b2 = stats.binned_statistic(np.log10(samples['period']),samples['lnprob'],statistic='sum',bins=50)
        ax[2].plot(a2[0:-1],hist2)
        ax[2].set_xlabel('log(Period)')
        ax[2].set_ylabel('Sum ln(Prob)')
        ax[2].set_xlim(xr)        
        fig.savefig(plotbase+'_samples.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_samples.png')
        
        # Plot best-fit model
        # one panel per band, mag vs. phase
        fig = plt.figure(figsize=(10,10))
        phase = (data['x']/bestperiod + bestoffset) % 1
        if hasattr(model, '__call__'):
            tmpl = model(phase)
        else:
            tmpl = np.interp(phase,template['x'],template['y'])            
        tphase = (np.linspace(0,1,100)+bestoffset) % 1
        si = np.argsort(tphase)
        tphase = tphase[si]
        if hasattr(model, '__call__'):            
            tmag = model(tphase)
        else:
            tmag = np.interp(tphase,template['x'],template['y'])
        model = tmag*bestamplitude+bestconstant
        dd = np.hstack((data['y'],model))
        yr = [np.max(dd)+0.05*dln.valrange(dd),np.min(dd)-0.30*dln.valrange(dd)]
        plt.plot(tphase,model,c='blue',zorder=1)        
        plt.errorbar(phase[ind],data['mag'][ind],yerr=data['err'][ind],c='gray',fmt='none',zorder=2)
        plt.scatter(phase[ind],data['mag'][ind],c='black',zorder=3)
        plt.annotate(txt,xy=(0.02,yr[1]+0.10*dln.valrange(dd)),ha='left')
        plt.xlabel('Phase')
        plt.ylabel('Magnitude')
        plt.xlim(0,1)
        plt.ylim(yr)
        plt.set_title('Period=%.3f  Offset=%.3f  Amplitude=%.3f  Constant=%.3f  ln(Prob)=%.3f' %
                      (bestperiod,bestoffset,bestamplitude,bestconstant,bestlnprob))
        fig.savefig(plotbase+'_best.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_best.png')    
 

        
#--------------------------------------------------------------------------------------------------------------------
        

class Sampler:
    """

    Generic sampler of periodic signals.

    args : tuple
       Must at least contain (x, y, yerr).  It can additional contain other positional
          arguments to be passed to log_probability().
    log_probability : function
       Function that calculates the ln probability given (theta, x, y, yerr).  It must also
         perform the marginalization over the non-linear parameters.
    kwargs : dict, optional
       Dictionary of keyword arguments to pass to log_probability() function.

    """
    
    def __init__(self,args,log_probability,kwargs=None):
        self._args = args
        # args should be (x,y,yerr, and other additional arguments to be passed to the functions)        
        self._log_probability = log_probability
        self._kwargs = kwargs
        # kwargs is a dictionary of additional keyword arguments to be passed to log_probability()

    def copy(self):
        """ Make a copy."""
        return copy.deepcopy(self)
        
    def run(self,pmin=0.1,pmax=None,minsample=128,npoints=200000):
        """ Run the sampling."""

        x = self._args[0]
        y = self._args[1]
        yerr = self._args[2]
        ndata = len(x)
        args = self._args
        kwargs = self._kwargs
        model = self._model
        log_probability = self._log_probability
        
        # Period range
        if pmax is None:
            pmax = (np.max(x)-np.min(x))*2
        lgminp = np.log10(pmin)
        lgmaxp = np.log10(pmax)
    
        print('Pmin = %.3f' % pmin)
        print('Pmax = %.3f' % pmax)    
        self._pmin = pmin
        self._pmax = pmax

        # Loop until we have enough samples
        nsamples,count = 0,0
        trials,samples = None,None
        dtt = [('period',float),('offset',float),('lnprob',float)]
        while (nsamples<minsample):
    
            # Uniformly sample from log(pmin) to log(pmax)
            period = np.random.rand(npoints)*(lgmaxp-lgminp)+lgminp    
            period = 10**period
            # Uniformly sample from 0 to 1
            offset = np.random.rand(npoints)

            # Calculate the ln probabilty
            lnprob = log_probability([period,offset],*args,**kwargs)

            # Save the information
            trials1 = np.zeros(npoints,dtype=dtt)
            trials1['period'] = period
            trials1['offset'] = offset
            trials1['lnprob'] = lnprob        
            if trials is None:
                trials = trials1
            else:
                trials = np.hstack((trials,trials1))
        
            # Rejection sampling
            draw = np.random.rand(npoints)
            ind, = np.where(draw < np.exp(lnprob))  
            if len(ind)>0:
                samp1 = np.zeros(len(ind),dtype=dtt)
                samp1['period'] = period[ind]
                samp1['offset'] = offset[ind]
                samp1['lnprob'] = lnprob[ind]
                if samples is None:
                    samples = samp1
                else:
                    samples = np.hstack((samples,samp1))
                nsamples += len(ind)
            
            print(count+1,nsamples)
            count += 1
            
        # Convert to astropy tables
        samples = Table(samples)
        trials = Table(trials)
        self.samples = samples
        self.trials = trials

        best = np.argmax(trials['lnprob'])
        bestperiod = trials['period'][best]
        bestoffset = trials['offset'][best]
        bestlnprob = trials['lnprob'][best]
        print('Best period = %.4f' % bestperiod)
        print('Best offset = %.4f' % bestoffset)
        print('Best lnprob = %.4f' % bestlnprob)
        self.bestperiod = bestperiod
        self.bestoffset = bestoffset
        self.bestlnprob = bestlnprob
        
        ntrials = npoints*count
        print('ntrials = ',ntrials)

        # If unimodal, run emcee
        medperiod = np.median(samples['period'])
        delta = (4*medperiod**2)/(2*np.pi*(np.max(x)-np.min(x)))
        deltap = medperiod**2/(2*np.pi*(np.max(x)-np.min(x)))
        rmsperiod = np.sqrt(np.mean((samples['period']-medperiod)**2))
        unimodal = False

        if rmsperiod < delta and unirefine:
            print('Unimodal PDF, finer sampling')
            unimodal = True
            # Fine sampling around the maximum
            rmsperiod = dln.mad(samples['period'].data-medperiod,zero=True)
            pmin2 = np.maximum(medperiod - 3*rmsperiod,pmin)
            pmax2 = medperiod + 3*rmsperiod
            medoffset = np.median(samples['offset'])
            rmsoffset = dln.mad(samples['offset'].data-medoffset,zero=True)
            offsetmin2 = np.maximum(medoffset-3*rmsoffset,offsetmin)
            offsetmax2 = medoffset+3*rmsoffset
            # Uniformly sample from min to max            
            period2 = np.random.rand(npoints)*(pmax2-pmin2)+pmin2
            offset2 = np.random.rand(npoints)*(offsetmax2-offsetmin2)+offsetmin2
            # Calculate the ln probabilty
            lnprob2 = log_probability([period2,offset2],*args,**kwargs)

            # Save trial information
            trials0 = trials
            del trials
            trials = np.zeros(npoints,dtype=dtt)
            trials['period'] = period2
            trials['offset'] = offset2
            trials['lnprob'] = lnprob2
            
            # Rejection sampling
            draw = np.random.rand(npoints)
            ind, = np.where(draw < np.exp(lnprob))                
            if len(ind)>0:
                # Creat table
                dt = [('period',float),('offset',float),('lnprob',float)]
                samples = np.zeros(len(ind),dtype=dt)
                samples['period'] = period2[ind]
                samples['offset'] = offset2[ind]
                samples['lnprob'] = lnprob2[ind] 

            samples = Table(samples)
            trials = Table(trials)
            self.samples = samples
            self.trials = trials

            # Get best values
            best1 = np.argmax(trials['lnprob'])
            bestperiod = trials['period'][best1]
            bestoffset = trials['offset'][best1]
            bestlnprob = trials['lnprob'][best1]

            if verbose:
                print('Best period = %.4f' % bestperiod)
                print('Best offset = %.4f' % bestoffset)
                print('Best lnprob = %.4f' % bestlnprob)

            self.bestperiod = bestperiod
            self.bestoffset = bestoffset
            self.bestlnprob = bestlnprob

        self.unimodal = unimodal

        # Construct best dictionary
        best = {'period':bestperiod,'phase':bestoffset,'lnprob':bestlnprob}    

        return samples, trials, best
            
            
    def plots(self,plotbase='sampler',bins=(200,200)):
        """ Make the plots."""

        x = self._args[0]
        y = self._args[1]
        yerr = self._args[2]
        ndata = len(x)
        bestperiod = self.bestperiod
        bestoffset = self.bestoffset
        bestlnprob = self.bestlnprob
        samples = self.samples
        trials = self.trials

        # Make plots
        matplotlib.use('Agg')
        fig,ax = plt.subplots(2,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        xr = [np.min(np.log10(trials['period'])),np.max(np.log10(trials['period']))]
        # 2D density map
        im,b,c,d = stats.binned_statistic_2d(trials['offset'],np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=bins)
        z1 = ax[0].imshow(im,aspect='auto',origin='lower',extent=(c[0],c[-1],b[0],b[-1]))
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        ax[0].set_xlim(xr)
        plt.colorbar(z1,ax=ax[0],label='Mean ln(Prob)')
        # Period histogram
        hist,a,b = stats.binned_statistic(np.log10(trials['period']),trials['lnprob'],statistic='mean',bins=1000)
        ax[1].plot(a[0:-1],hist)
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Mean ln(Prob)')
        ax[1].set_xlim(xr)        
        fig.savefig(plotbase+'_trials.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_trials.png')
    
        # Plot offset vs. period color-coded by lnprob
        fig,ax = plt.subplots(2,1,constrained_layout=True)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        # Plot offset vs. period color-coded by lnprob
        z1 = ax[0].scatter(np.log10(samples['period']),samples['offset'],c=samples['lnprob'])
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        plt.colorbar(z1,ax=ax[0],label='ln(Prob)')
        # Sum of lnprob
        hist2,a2,b2 = stats.binned_statistic(np.log10(samples['period']),samples['lnprob'],statistic='sum',bins=200)
        ax[1].plot(a2[0:-1],hist2)
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Sum ln(Prob)')
        fig.savefig(plotbase+'_samples.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_samples.png')
        
