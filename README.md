# ssm-analysis Overview

A set of tools for analyzing outputs from the Salish Sea Model.

* `SSM_Grid/*`: 2-D Mesh files defining the model grid.
* `gis/ssm domain utm.*`: a manually created shapefile defining the analysis
  domain.
* `gis/ecology masked cells.*`: a manually created shapefile exported from
  [Ecology's GDB](https://fortress.wa.gov/ecy/ezshare/EAP/SalishSea/SalishSeaModelBoundingScenarios.html).
* `gis/ssm domain nodes.*`: a shapefile defining the polygon for each node
  within the analysis domain.
* `ProcessGrid.ipynb`: A notebook that produces `ssm domain nodes.shp` and
  related shapefiles using the 2-D Mesh model grid and `ssm domain utm.*`.
* `ssmhist2cdf/`: A C++ program that parses `ssm_history_*` text output
  files into a NetCDF file that's easier to process.
* `do_rawcdf_extraction.py`: A script that extracts from NetCDF files only a
  subset of nodes for a single water quality variable at the bottom layer,
  and writes to a variable in a NetCDF file. This script was intended for
  extracting bottom DO and has been replaced.
* `rawcdf_extract.py`: A more complete extraction script that can extract
  a subset of nodes for various vertical slices of the water column.
  Supports node masking.
* `DO ecology extraction.ipynb`: A notebook that extracts DO data for the
  analysis domain from WA Ecology's NetCDF model result files.
* `DO Compliance.ipynb`: A notebook that does a simple analysis of dissolved
  oxygen results from existing and reference condition runs.
* `validation/`: A set of notebooks that can validate model outputs against
  observations in a database.
* `Bottom DO.ipynb`: A notebook that produces comparison plots of bottom
  dissolved oxygen from multiple model runs.

# The Python environment

A conda environment export is included here with all the needed dependencies.
To install use `conda env create -f environment.yml`.

One more thing you need to do for a few of the scripts and notebooks to work
is add the `fvcom_util` path to `lib/python3.8/site-packages/conda.pth` inside
your new conda environment. Eventually I plan on breaking that code off into
its own Python module.

# A typical workflow

Create a shape in a GIS program that encircles all of the nodes of interest for
an analysis. See `gis/ssm domain utm.shp` for an example.

Run the `ProcessGrid.ipynb` notebook using the above shapefile to generate
new shapefiles for all of the contained nodes, tracer areas, and masked cells.

If you are running FVCOMICM v2, you'll need to use `ssmhist2cdf` to convert
history file output to NetCDF format. After you run FVCOMICM v4 (or FVCOM),
you'll have NetCDF output files without having to do anything else.

Validation is performed using the notebooks in `validation/` against the full
NetCDF output files.

Visualizations are easier to run against a small extraction of the full output
files. To make those extractions, use the script `rawcdf_extract.py`.

Finally, run one or more of the visualization notebooks by pointing them at
the NetCDF file made by `rawcdf_extract.py`.

# rawcdf extract

The script takes three required sets of arguments: all the model output NetCDF
files, the name of a NetCDF file to write extracted data to, and a prefix
string to use for the extracted variable names. There are a few other arguments
that can be passed as options which will commonly be desired:

* To use a custom domain area, use `-d`. The default is my domain shapefile
  for Puget Sound. See also the masked nodes that can be specified with `-m`.
  These only have to be specified the first time the script is run to create a
  new extraction NetCDF; later runs will read the existing NetCDF file to know
  which nodes to extract.
* By default the script only extracts bottom-layer dissolved oxygen. To extract
  a different variable and/or a different vertical slice, use the option
  `--invar` once for each variable to extract. This option expects a string
  of the format `VARNAME:sliceattr,sliceattr,...` where VARNAME is the variable
  name to extract from the model output (such as `DOXG`, `NO3`, `B1`, ...) and
  sliceattr is as many of the following attribute identifiers needed to specify
  the extraction: `bottom`, `max`, `min`, `mean`, `photic`, `all`. `all`
  extracts all depth layers and the others will result in a collapsed variable
  that has no depth dimension. `bottom` extracts only the bottom layer. `max`,
  `mean`, and `min` compute a single result for the entire water column.
  `photic` will restrict extraction to only the photic zone (requires the
  `IAVG` light variable to be present). This one typically needs another
  attribute specified alongside it, so use something like `mean,photic` to get
  the mean of just the photic zone. Not all combinations will make sense, and
  there's not much error checking so tacking together a bunch of conflicting
  attributes is not recommended.

The script can be run multiple times to extract variables from multiple
different runs into the same file. This is useful to assemble results from
different runs in an experiment, then they can all be plotted together. Just
use different prefixes to refer to different runs. If for some reason you want
to replace an extracted result variable in a file, keeping the same prefix
(for instance because the model run was repeated after a mistake) you will
need to add the option `--force-overwrite`.

Here's the full help message from running `--help`:

```
usage: rawcdf_extract.py [-h] [-d DOMAIN_NODE_SHAPEFILES]
                         [-m MASKED_NODES_FILE] [--invar INPUT_VARS] [-v]
                         [-c CHUNK_SIZE] [--cache] [--force-overwrite]
                         incdf [incdf ...] outcdf outprefix

Extract data from SSM netcdf output files

positional arguments:
  incdf                 each input CDF file
  outcdf                the output CDF file (created if it doesn't exist)
  outprefix             a prefix for the extracted variables in the output CDF

optional arguments:
  -h, --help            show this help message and exit
  -d DOMAIN_NODE_SHAPEFILES
                        Specify a domain node shapefile
  -m MASKED_NODES_FILE  Specify a different masked nodes text file
  --invar INPUT_VARS    Extract the values of a different output variable
  -v, --verbose         Print progress messages during the extraction
  -c CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Process this many CDF files at once
  --cache               Use a read/write cache in a temporary directory
  --force-overwrite     Force overwriting of an existing output variable
```

## Examples

Extract bottom dissolved oxygen from a model run:

```
rawcdf_extract.py /path/to/model/outputs/ssm_FVCOMICM_000*.nc
extracted_output.nc myrun
```

Extract mean dissolved oxygen from a model run:

```
rawcdf_extract.py --invar DOXG:mean /path/to/model/outputs/ssm_FVCOMICM_000*.nc
extracted_output_2.nc myrun
```

Extract mean photic zone phytoplankton and minimum photic zone nitrate:

```
rawcdf_extract.py --invar B1:mean,photic --invar B2:mean,photic --invar
NO3:min,photic /path/to/model/outputs/ssm_FVCOMICM_000*.nc
extracted_output_3.nc myrun
```
