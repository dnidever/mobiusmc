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


def solvevariable(data,template,ampratios,bandindex,period,offset,totwtdict,totwtydict):
    """ Solve variable star for a single period/phase value."""
    
    ndata = len(data)
        
    # Calculate phase for each data point
    phase = (data['jd']/period + offset) % 1
            
    # Calculate template values for this set of period and phase
    tmpl = np.interp(phase,template['phase'],template['mag'])
            
    # -- Find best fitting values for linear parameters ---
    # Calculate amplitude
    # term1 = Sum of XY
    # term2 = Sum of X * Y / W 
    # term3 = Sum of X^2
    # term4 = Sum of X * X / W
    # amplitude = (term1 - term2)/(term3 - term4)
    term1 = 0.0
    term2 = 0.0
    term3 = 0.0
    term4 = 0.0
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
            
    # Calculate best mean magnitudes
    # mean mag = (Y - amplitude * X)/W
    meanmag = {}
    for b in bandindex.keys():
        meanmag1 = (totwtydict[b] - amplitude * totwtxdict[b])/totwtdict[b]
        meanmag[b] = meanmag1
            
    # Calculate likelihood/chisq
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

    return amplitude,meanmag,model,lnlkhood


def log_likelihood_variable(theta,x,y,err,data=None,template=None,ampratios=None,bandindex=None,totwtdict=None,totwtydict=None,**kwargs):

    try:
        dum = data['wt']
    except:
        data = Table(data)
        data['wt'] = 1/data['err']**2
    
    if bandindex is None:
        # Get band index
        uband = np.unique(data['band'])
        nband = len(uband)
        bandindex = {}
        for i,b in enumerate(uband):
            ind, = np.where(data['band']==b)
            bandindex[b] = ind     
    if totwtdict is None or totwtydict is None:
        # Pre-calculate some terms that are constant
        totwtdict = {}
        totwtydict = {}
        for b in uband:
            ind = bandindex[b]
            totwtdict[b] = np.sum(data['wt'][ind])
            totwtydict[b] = np.sum(data['wt'][ind] * data['mag'][ind])
            
    
    period, offset = theta
    ndata = len(data)
    
    # Calculate phase for each data point
    if period.size > 1:
        # Get phase and template points
        phase = (data['jd'].reshape(-1,1)/period.reshape(1,-1) + offset.reshape(1,-1)) % 1
        tmpl = np.interp(phase.ravel(),template['phase'],template['mag'])
        tmpl = tmpl.reshape(ndata,period.size)
    else:
        phase = (data['jd']/period + offset) % 1            
        # Calculate template values for this set of period and phase
        tmpl = np.interp(phase,template['phase'],template['mag'])
            
    # -- Find best fitting values for linear parameters ---
    # Calculate amplitude
    # term1 = Sum of XY
    # term2 = Sum of X * Y / W 
    # term3 = Sum of X^2
    # term4 = Sum of X * X / W
    # amplitude = (term1 - term2)/(term3 - term4)
    term1 = 0.0
    term2 = 0.0
    term3 = 0.0
    term4 = 0.0
    totwtxdict = {}
    for b in bandindex.keys():
        ind = bandindex[b]
        if period.size > 1:
            totwtx1 = np.sum(data['wt'][ind].reshape(-1,1) * tmpl[ind,:]*ampratios[b],axis=0)
            totwtx2 = np.sum(data['wt'][ind].reshape(-1,1) * (tmpl[ind,:]*ampratios[b])**2,axis=0)
            totwtxy = np.sum(data['wt'][ind].reshape(-1,1) * tmpl[ind,:]*ampratios[b] * data['mag'][ind].reshape(-1,1),axis=0)
        else:
            totwtx1 = np.sum(data['wt'][ind] * tmpl[ind]*ampratios[b])
            totwtx2 = np.sum(data['wt'][ind] * (tmpl[ind]*ampratios[b])**2)
            totwtxy = np.sum(data['wt'][ind] * tmpl[ind]*ampratios[b] * data['mag'][ind])                  
        totwtxdict[b] = totwtx1
        term1 += totwtxy
        term2 += totwtx1 * totwtydict[b] / totwtdict[b]
        term3 += totwtx2
        term4 += totwtx1**2 / totwtdict[b]
    amplitude = (term1-term2)/(term3-term4)
            
    # Calculate best mean magnitudes
    # mean mag = (Y - amplitude * X)/W
    meanmag = {}
    for b in bandindex.keys():
        meanmag1 = (totwtydict[b] - amplitude * totwtxdict[b])/totwtdict[b]
        meanmag[b] = meanmag1
            
    # Calculate likelihood/chisq
    if period.size > 1:
        model = np.zeros((ndata,period.size),float)
        resid = np.zeros((ndata,period.size),float)
        wtresid = np.zeros((ndata,period.size),float)
        for b in uband:
            ind = bandindex[b]
            model1 = tmpl[ind,:]*ampratios[b]*amplitude+meanmag[b]
            model[ind,:] = model1
            resid[ind,:] = data['mag'][ind].reshape(-1,1)-model1
            wtresid[ind,:] = resid[ind,:]**2 * data['wt'][ind].reshape(-1,1)
        lnlikelihood = -0.5*np.sum(wtresid,axis=0)
        lnlikelihood += -0.5*np.sum(np.log(2*np.pi*data['err']**2))
    else:
        model = np.zeros(ndata,float)
        resid = np.zeros(ndata,float)
        wtresid = np.zeros(ndata,float)        
        for b in bandindex.keys():
            ind = bandindex[b]
            model1 = tmpl[ind]*ampratios[b]*amplitude+meanmag[b]
            model[ind] = model1
            resid[ind] = data['mag'][ind]-model1
            wtresid[ind] = resid[ind]**2 * data['wt'][ind]
        lnlikelihood = -0.5*np.sum(wtresid + np.log(2*np.pi*data['err']**2))

    return lnlikelihood

def model_variable(phase,**kwargs):
    """ Generate variable star template model using phase."""
    template = kwargs['template']
    tmpl = np.interp(phase.ravel(),template['phase'],template['mag'])
    if phase.ndim > 1:
        tmpl = tmpl.reshape(phase.shape)
    return tmpl


#def log_likelihood(theta, x, y, yerr):
#    m, b, log_f = theta
#    model = m * x + b
#    sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
#    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_variable(theta,prange=None):
    period = theta[0]
    pmin = np.min(period)
    pmax = np.max(period)
    lnprior = np.log(1/(1.0*(np.log10(pmax)-np.log10(pmin))))
    return lnprior

def log_probability_variable(theta, x, y, yerr, *args, **kwargs):
    lp = log_prior_variable(theta)
    #if not np.isfinite(lp):
    #    return -np.inf
    return lp + log_likelihood_variable(theta, x, y, yerr, *args, **kwargs)


class VariableSampler:
    """
    Class for doing sampling of variable star lightcurve ddata.

    Parameters
    ----------
    catalog : table
       Catalog of data points, just have mag, err, jd, band
    template : table
       Template information.  Need phase and mag columns.
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

        # filter not band input
        if 'band' not in self.data.colnames and 'filter' in self.data.colnames:
            self.data['band'] = self.data['filter']
        # mjd not jd input
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


            # Get phase and template points
            phase = (data['jd'].reshape(-1,1)/period.reshape(1,-1) + offset.reshape(1,-1)) % 1
            tmpl = np.interp(phase.ravel(),template['phase'],template['mag'])
            tmpl = tmpl.reshape(ndata,npoints)
            
            # -- Find best fitting values for linear parameters ---
            # Calculate amplitude
            # term1 = Sum of XY
            # term2 = Sum of X * Y / W 
            # term3 = Sum of X^2
            # term4 = Sum of X * X / W
            # amplitude = (term1 - term2)/(term3 - term4)
            term1,term2,term3,term4 = 0,0,0,0
            totwtxdict = {}
            for b in uband:
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
    
            # Calculate best mean magnitudes
            # mean mag = (Y - amplitude * X)/W
            meanmag = {}
            for b in uband:
                meanmag1 = (totwtydict[b] - amplitude * totwtxdict[b])/totwtdict[b]
                meanmag[b] = meanmag1
            
            # Calculate likelihood/chisq
            model = np.zeros((ndata,npoints),float)
            resid = np.zeros((ndata,npoints),float)
            wtresid = np.zeros((ndata,npoints),float)        
            for b in uband:
                ind = bandindex[b]
                model1 = tmpl[ind,:]*ampratios[b]*amplitude+meanmag[b]
                model[ind,:] = model1
                resid[ind,:] = data['mag'][ind].reshape(-1,1)-model1
                wtresid[ind,:] = resid[ind,:]**2 * data['wt'][ind].reshape(-1,1)
            lnlikelihood = -0.5*np.sum(wtresid,axis=0)
            lnlikelihood += -0.5*np.sum(np.log(2*np.pi*data['err']**2))

            # Calculate ln probability = ln prior + ln likelihood
            # use flat prior, divide by area
            lnprior = np.ones(npoints,float) + np.log(1/(1.0*(lgmaxp-lgminp)))
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
    
        
        # If unimodal, run emcee
        medperiod = np.median(samples['period'])
        delta = (4*medperiod**2)/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        deltap = medperiod**2/(2*np.pi*(np.max(data['jd'])-np.min(data['jd'])))
        rmsperiod = np.sqrt(np.mean((samples['period']-medperiod)**2))
        unimodal = False
        if rmsperiod < delta and unirefine:
            print('Unimodal PDF, finer sampling')
            unimodal = True

            # Do NOT run emcee, just redo sampling with a tighter range on
            # the unimodal region

            import pdb; pdb.set_trace()
            
            # Run this method again with unirefine=False (otherwise infinite loop)
            pmin2 = np.maximum(medperiod - 5*rmsperiod,0)
            pmax2 = medperiod + 5*rmsperiod
            medperiod = np.median(samples['offset'])
            rmsperiod = np.sqrt(np.mean((samples['offset']-medoffset)**2))
            offsetmin2 = np.maximum(medoffset-5*rmsoffset,0)
            offsetmax2 = medoffset+5*rmsoffset
            samples2,trial2,best2 = self.run(pmin2,pmax2,offsetrange=[offsetmin2,offsetmax2],
                                             minsample=1024,npoints=200000,unirefine=False,
                                             verbose=False)

            import pdb; pdb.set_trace()

            # The maximum likelihood parameters
            bestind = np.unravel_index(np.argmax(emsampler.lnprobability),emsampler.lnprobability.shape)
            pars_ml = emsampler.chain[bestind[0],bestind[1],:]
        
            labels = ['Period','Offset']
            for i in range(ndim):
                mcmc = np.percentile(emsamples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                print(r'%s = %.3f -%.3f/+%.3f' % (labels[i], pars_ml[i], q[0], q[1]))
        
            #fig = corner.corner(emsamples, labels=['Period','Offset'])
            #plt.savefig(plotbase+'_corner.png',bbox_inches='tight')
            #plt.close(fig)
            #print('Corner plot saved to '+plotbase+'_corner.png')
    
            bestperiod = pars_ml[0]
            bestoffset = pars_ml[1]
            bestlnprob = emsampler.lnprobability[bestind[0],bestind[1]]
            bestamplitude,bestmeanmag,model,lnlkhood = solvevariable(data,template,ampratios,bandindex,
                                                                     bestperiod,bestoffset,totwtdict,totwtydict)

            
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
        best = {'period':bestperiod,'phase':bestoffset,'amplitude':bestamplitude,'meanmag':bestmeanmag,'lnprob':bestlnprob}
        
        #import pdb; pdb.set_trace()

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
        # Plot offset vs. period color-coded by lnprob
        z1 = ax[0].scatter(np.log10(samples['period']),samples['offset'],c=samples['lnprob'])
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        plt.colorbar(z1,ax=ax[0],label='ln(Prob)')
        # Plot amplitude vs. period color-coded by lnprob
        z2 = ax[1].scatter(np.log10(samples['period']),samples['amplitude'],c=samples['lnprob'])
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Amplitude')
        plt.colorbar(z2,ax=ax[1],label='ln(Prob)')
        # Sum of lnprob
        hist2,a2,b2 = stats.binned_statistic(np.log10(samples['period']),samples['lnprob'],statistic='sum',bins=200)
        ax[2].plot(a2[0:-1],hist2)
        ax[2].set_xlabel('log(Period)')
        ax[2].set_ylabel('Sum ln(Prob)')
        fig.savefig(plotbase+'_samples.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_samples.png')
        
        # Plot best-fit model
        # one panel per band, mag vs. phase
        fig,ax = plt.subplots(nband,1)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        phase = (data['jd']/bestperiod + bestoffset) % 1
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
                ax[i].set_title('Period=%.3f  Offset=%.3f  Amplitude=%.3f  ln(Prob)=%.3f' % (bestperiod,bestoffset,bestamplitude,bestlnprob))
        fig.savefig(plotbase+'_best.png',bbox_inches='tight')
        plt.close(fig)
        print('Saving to '+plotbase+'_best.png')    

#--------------------------------------------------------------------------------------------------------------------


class LinearModelSampler:
    """
    Class for doing sampling of periodic linear model (y=a*model(phase)+b)

    Parameters
    ----------
    data : table
       Catalog of data points, just have x, y, yerr.
    model : function
       Model function or template with x and y columns.
    minerror : float, optional
       Minimum error to use.  Default is 0.02.
    
    """

    def __init__(self,data,model,minerror=0.02):

        # Create the sampling for Period (pmin to pmax) and phase offset (0-1)

        self.data = Table(data).copy()
        for n in self.data.colnames:
            self.data[n].name = n.lower()   # change columns names to lower case
        self.model = model
        
        # Add weights to internal catalog
        self.data['wt'] = 1/np.maximum(self.data['yerr'],minerror)**2
        data = self.data
    
        print(str(ndata)+' data points')
        print('time baseline = %.2f' % (np.max(data['jd'])-np.min(data['jd'])))
                
        # Pre-calculate some terms that are constant
        totwt = np.sum(data['wt'])
        totwty = np.sum(data['wt']*data['y'])
        self._totwt = totwt
        self._totwty = totwty

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


            # Get phase and template points
            phase = (data['x'].reshape(-1,1)/period.reshape(1,-1) + offset.reshape(1,-1)) % 1
            if hasattr(model, '__call__'):            
                tmpl = model(phase.ravel())
            else:
                tmpl = np.interp(phase.ravel(),model['x'],model['y'])
            tmpl = tmpl.reshape(ndata,npoints)
            
            # -- Find best fitting values for linear parameters ---
            # Calculate amplitude
            # term1 = Sum of XY
            # term2 = Sum of X * Y / W 
            # term3 = Sum of X^2
            # term4 = Sum of X * X / W
            # amplitude = (term1 - term2)/(term3 - term4)
            term1,term2,term3,term4 = 0,0,0,0
            totwtx = np.sum(data['wt'].reshape(-1,1) * tmpl,axis=0)
            totwtx2 = np.sum(data['wt'].reshape(-1,1) * tmpl**2,axis=0)
            totwtxy = np.sum(data['wt'].reshape(-1,1) * tmpl * data['y'].reshape(-1,1),axis=0)      
            term1 += totwtxy
            term2 += totwtx * totwty / totwt
            term3 += totwtx2
            term4 += totwtx**2 / totwt
            amplitude = (term1-term2)/(term3-term4)
    
            # Calculate cpmstant offfset
            # mean mag = (Y - amplitude * X)/W
            constant = (totwty-amplitude*totwtx)/totwt
            
            # Calculate likelihood/chisq
            model = np.zeros((ndata,npoints),float)
            resid = np.zeros((ndata,npoints),float)
            wtresid = np.zeros((ndata,npoints),float)        
            model1 = tmpl[ind,:]*ampratios[b]*amplitude+meanmag[b]
            resid = data['y'].reshape(-1,1)-model1
            wtresid = resid**2 * data['wt'].reshape(-1,1)
            lnlikelihood = -0.5*np.sum(wtresid,axis=0)
            lnlikelihood += -0.5*np.sum(np.log(2*np.pi*data['yerr']**2))

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

            # Do NOT run emcee, just redo sampling with a tighter range on
            # the unimodal region

            import pdb; pdb.set_trace()
            
            # Run this method again with unirefine=False (otherwise infinite loop)
            pmin2 = np.maximum(medperiod - 5*rmsperiod,0)
            pmax2 = medperiod + 5*rmsperiod
            medperiod = np.median(samples['offset'])
            rmsperiod = np.sqrt(np.mean((samples['offset']-medoffset)**2))
            offsetmin2 = np.maximum(medoffset-5*rmsoffset,0)
            offsetmax2 = medoffset+5*rmsoffset
            samples2,trial2,best2 = self.run(pmin2,pmax2,offsetrange=[offsetmin2,offsetmax2],
                                             minsample=1024,npoints=200000,unirefine=False,
                                             verbose=False)

            import pdb; pdb.set_trace()

            # The maximum likelihood parameters
            bestind = np.unravel_index(np.argmax(emsampler.lnprobability),emsampler.lnprobability.shape)
            pars_ml = emsampler.chain[bestind[0],bestind[1],:]
        
            labels = ['Period','Offset']
            for i in range(ndim):
                mcmc = np.percentile(emsamples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                print(r'%s = %.3f -%.3f/+%.3f' % (labels[i], pars_ml[i], q[0], q[1]))
        
            #fig = corner.corner(emsamples, labels=['Period','Offset'])
            #plt.savefig(plotbase+'_corner.png',bbox_inches='tight')
            #plt.close(fig)
            #print('Corner plot saved to '+plotbase+'_corner.png')
    
            bestperiod = pars_ml[0]
            bestoffset = pars_ml[1]
            bestlnprob = emsampler.lnprobability[bestind[0],bestind[1]]
            bestamplitude,bestconstant,tmpl,lnlkhood = solvelinearmodel(data,model,bestperiod,bestoffset,totwt,totwty)

            
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
        best = {'period':bestperiod,'phase':bestoffset,'amplitude':bestamplitude,'consant':bestconstant,'lnprob':bestlnprob}
        
        #import pdb; pdb.set_trace()

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
        # Plot offset vs. period color-coded by lnprob
        z1 = ax[0].scatter(np.log10(samples['period']),samples['offset'],c=samples['lnprob'])
        ax[0].set_xlabel('log(Period)')
        ax[0].set_ylabel('Phase Offset')
        plt.colorbar(z1,ax=ax[0],label='ln(Prob)')
        # Plot amplitude vs. period color-coded by lnprob
        z2 = ax[1].scatter(np.log10(samples['period']),samples['amplitude'],c=samples['lnprob'])
        ax[1].set_xlabel('log(Period)')
        ax[1].set_ylabel('Amplitude')
        plt.colorbar(z2,ax=ax[1],label='ln(Prob)')
        # Sum of lnprob
        hist2,a2,b2 = stats.binned_statistic(np.log10(samples['period']),samples['lnprob'],statistic='sum',bins=200)
        ax[2].plot(a2[0:-1],hist2)
        ax[2].set_xlabel('log(Period)')
        ax[2].set_ylabel('Sum ln(Prob)')
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

    log_probability : function
       Function that calculates the ln probability given (theta, x, y, err).  It must also
         perform the marginalization over the non-linear parameters.
    args : tuple
       Must at least contain (x, y, err).  It can additional contain other positional
          arguments to be passed to log_probability().
    kwargs : dict, optional
       Dictionary of keyword arguments to pass to log_probability() function.

    """
    
    def __init__(self,log_probability,args=None,kwargs=None):
        self._log_probability = log_probability
        self._args = args
        # args should be (x,y,err, and other additional arguments to be passed to the functions)        
        self._kwargs = kwargs
        # kwargs is a dictionary of additional keyword arguments to be passed to log_probability()

    def copy(self):
        """ Make a copy."""
        return copy.deepcopy(self)
        
    def run(self,pmin=0.1,pmax=None,minsample=128,npoints=200000):
        """ Run the sampling."""

        x = self._args[0]
        y = self._args[1]
        err = self._args[2]
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
        if rmsperiod < delta:
            print('Unimodal PDF, running emcee')
            unimodal = True
            medoffset = np.median(samples['offset'])
            rmsoffset = np.sqrt(np.mean((samples['offset']-medoffset)**2))
            # Set up the MCMC sampler
            ndim, nwalkers = 2, 10
            delta = [rmsperiod*5,rmsoffset*5]
            initpar = [bestperiod, bestoffset]
            pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]
            emsampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                              args=args,kwargs=kwargs)
            steps = 100
            out = emsampler.run_mcmc(pos, steps)
            emsamples = emsampler.chain[:, np.int(steps/2):, :].reshape((-1, ndim))

            # The maximum likelihood parameters
            bestind = np.unravel_index(np.argmax(emsampler.lnprobability),emsampler.lnprobability.shape)
            pars_ml = emsampler.chain[bestind[0],bestind[1],:]
        
            labels = ['Period','Offset']
            for i in range(ndim):
                mcmc = np.percentile(emsamples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                print(r'%s = %.3f -%.3f/+%.3f' % (labels[i], pars_ml[i], q[0], q[1]))
    
            bestperiod = pars_ml[0]
            bestoffset = pars_ml[1]
            bestlnprob = emsampler.lnprobability[bestind[0],bestind[1]]

            print('Best period = %.4f' % bestperiod)
            print('Best offset = %.4f' % bestoffset)
            print('Best lnprob = %.4f' % bestlnprob)

            self.bestperiod = bestperiod
            self.bestoffset = bestoffset
            self.bestlnprob = bestlnprob

        self.unimodal = unimodal

        return samples, trials
            
            
    def plots(self,plotbase='sampler',bins=(200,200)):
        """ Make the plots."""

        x = self._args[0]
        y = self._args[1]
        err = self._args[2]
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
        
