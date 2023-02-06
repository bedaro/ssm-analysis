#ifndef _COPYFROMHYD_H
#define _COPYFROMHYD_H

#include <netcdf>

/**
 * Copies important grid data from a FVCOM NetCDF output file
 */
void copyFromHyd(const netCDF::NcFile& ncHyd, netCDF::NcFile& ncFile);

/**
 * Copies the contents of a variable, including attributes, to a new
 * NetCDF file.
 */
void copyVariable(const netCDF::NcVar& var, netCDF::NcFile& dest);

#endif
