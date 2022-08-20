`extract_sections.py` creates a NetCDF file for each section defined in a .ini
file with arrays of hourly transport and tracer values on the section,
arranged as (t, z, ele). It is meant to mimic Dr. Parker MacCready's script
of the same name in [LiveOcean's x_tef](https://github.com/parkermac/LiveOcean/tree/master/x_tef)
and has many of the same options.

Key operational differences are:
 * sections are specified in a .ini file read with Python's ConfigParser
 * output file dimension for depth is labeled `s_z` instead of `s_rho`
 * transects do not have to be vertical or horizontal; they can go in any
  direction on FVCOM's unstructured grid.

Much of the code is also based on a rewrite of [Ted Conroy's version](https://github.com/tedconroy/ocean-model-codes/blob/master/fvcom/fvcom_calcfluxsect.m)
of an FVCOM transport calculator Matlab script. The original was written by
David Ralston (WHOI).

There is also a bit of similarity to my `rawcdf_extract.py` script in this
repository.

# Usage

The script has three required arguments, in order:

 * One or more FVCOM output NetCDF files (can use a wildcard)
 * The path to a config file that defines sections to extract (see below)
 * The path to an output directory (will be created if it does not exist) to
   store all the outputs.

Run the script with `--help` to see all of the many options. Important ones
are:

 * `--make-plots` will generate one map of all of the defined transects,
   showing directional arrows at each element center to illustrate the
   positive direction of flow (convention is for this to be up-estuary);
   and one profile plot for each transect showing a (flattened) cross section.
   This is a helpful diagnostic to ensure the transects have been properly set
   up and the code was able to successfully calculate all the correct segment
   coordinates.
 * `--output-start-date` sets the date corresponding to model time 0, in
   `YYYY.MM.DD` format. Unlike with LiveOcean, this information cannot be
   determined from an FVCOM output file and must be passed in. The sensible
   default is Jan 1, 2014.
 * `--date-string0` and `--date-string1` are optional, and can be used to
   restrict the extraction date range. By default the entire output timespan
   is extracted.

Once the NetCDF files are generated, they should be directly usable as inputs
to the rest of Parker's TEF processing toolset starting with
`process_sections.py`.

# Defining Sections

The config file follows the formatting conventions of
[Python's ConfigParser](https://docs.python.org/3/library/configparser.html),
which is close to Windows's INI files. For each transect, define a section in
the file and give it a descriptive name. There is only one required key,
`waypoints`, which is a list of at least two elements separated by spaces. The
shortest path connecting all the given waypoints in order will be used to
select the complete set of elements.

By default, positive flow is defined as going to the right of the direction the
transect's elements are specified. This can be seen on the overview plot. If
you need to flip the direction, provide the key `upesty` (for direction of
up-estuary) and give it the value `l` for left.

A short example:

```
[AdmInlet1]
waypoints = 7823 7663

[AdmInlet2]
waypoints = 10054 10063

[AdmInlet3]
waypoints = 12384 12817
```
