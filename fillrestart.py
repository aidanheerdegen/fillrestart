#!/usr/bin/env python
import argparse
from collections import ChainMap
from itertools import product
from pathlib import Path
import sys

from xarray.backends.api import to_netcdf
import xarray as xr
import cf_xarray as cfxr

def crude_fill(da, flip=True):
    # Do a crude fill forward and backward on all dimensions
    if flip:
        dim1 = da.cf['X'].name
        dim2 = da.cf['Y'].name
    else:
        dim1 = da.cf['Y'].name
        dim2 = da.cf['X'].name
    return da.interpolate_na(dim1).bfill(dim1).ffill(dim1).bfill(dim2).ffill(dim2)

def filter_time(ds):
    # Select only one first time slice
    if 'T' in ds.cf:
        ds = ds.cf.isel(T=0)
    return ds

def parse_args(arglist):
    """
    Parse arguments given as list (arglist)
    """
    parser = argparse.ArgumentParser(description="Fill missing data points in MOM model restart file for use with altered bathymetry")
    parser.add_argument("-v","--verbose", help="Verbose output", action='store_true')
    parser.add_argument("--flip", help="Flip order of filling (default is fill by longitude first)", action='store_false')
    parser.add_argument("-m","--maskfile", help="Specify file to use to mask out land cells from restart file", required=True)
    parser.add_argument("--zaxis", help="Name of the z-axis coordinate") 
    parser.add_argument("--maskvar", help="Name of variable to use as mask (default is first found)", default=None)
    parser.add_argument("--masklb", help="Lower bound above which value is a valid pixel in mask variable", type=float, default=0.)
    parser.add_argument("--maskub", help="Upper bound below which value is a valid pixel in mask variable", type=float, default=99999)
    parser.add_argument("input", help="MOM restart file to fill")
    parser.add_argument("output", help="Filename for filled file")

    return parser.parse_args(arglist)

def main_parse_args(arglist):
    """
    Call main with list of arguments. Callable from tests
    """
    # Must return so that check command return value is passed back to calling routine
    # otherwise py.test will fail
    return main(parse_args(arglist))

def main_argv():
    """
    Call main and pass command line arguments. This is required for setup.py entry_points
    """
    main_parse_args(sys.argv[1:])

def main(args):
    
    ds = xr.open_dataset(args.input)

    ds_mask = filter_time(xr.open_dataset(args.maskfile))

    if args.maskvar is None:
        maskvar = ds_mask[ds_mask.data_vars[0]]
    else:
        maskvar = ds_mask[args.maskvar]

    # Make mask
    mask = xr.where((maskvar > args.masklb) & (maskvar < args.maskub), True, False)

    for varname in ds.data_vars:
        print('Filling {}'.format(varname))
        var = ds[varname]

        slices = []
        if 'Z' in var.cf:
            zaxis = var.cf.axes['Z'][0]
            slices = [ {zaxis: z} for z in var[zaxis] ]
        if 'T' in var.cf:
            taxis = var.cf.axes['T'][0]
            slices = product(slices, [ {taxis: t} for t in var[taxis] ])

        for s in slices:
            # Combine array of dicts (slices) into single dict
            slicedict = ChainMap(*s) 
            # Pull out slices relevant to the mask, which likely has few dimensions
            maskdict = {k: slicedict[k] for k in mask.dims if k in slicedict}
            # Fill slice and save back into dataset
            ds[varname].loc[slicedict] = crude_fill(var.loc[slicedict].where(mask.loc[maskdict]), flip=args.flip)
        
        ds.to_netcdf(args.output)

if __name__ == "__main__":

    main_argv()
