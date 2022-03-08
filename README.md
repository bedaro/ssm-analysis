# ssm-analysis

A set of tools for analyzing outputs from the Salish Sea Model.

* `SSM_Grid/*`: 2-D Mesh files defining the model grid.
* `gis/ssm domain utm.*`: a manually created shapefile defining the analysis
  domain.
* `gis/ecology masked cells.*`: a manuall created shapefile exported from
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
  a subset of nodes for either the bottom layer or the complete water column.
  Additional depth aggregations, like photic zone, are intended to be
  implemented in the future. Supports node masking.
* `DO ecology extraction.ipynb`: A notebook that extracts DO data for the
  analysis domain from WA Ecology's NetCDF model result files.
* `DO Compliance.ipynb`: A notebook that does a simple analysis of dissolved
  oxygen results from existing and reference condition runs.
