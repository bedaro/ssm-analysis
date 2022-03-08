This collection of notebooks allows converting the text-based input files
of FVCOM 2.7 and FVCOM-ICM to and from Excel or NetCDF formats, which are
easier to manipulate in bulk.

I will document these more throughly in the future. Notebooks whose names
begin with `fvcom27_` are for converting FVCOM 2.7 input files, and `ssm_`
are for FVCOM-ICM input files. `*_read_*` notebooks read a text-based
input file and produce a NetCDF or Excel file, while `*_write_*` notebooks
read the file produced by the similarly named `*_read_*` notebook and convert
it back to the native text-based format. One can run the `read` notebook to
make a NetCDF or Excel file, manipulate it arbitrarily (with a custom
notebook or by hand), and then run the `write` notebook on the modification
to easily make input files for various scenarios.

Examples of these files are available from
[SSMC](https://gitlab.com/ssmc/fvcom-icm4.0) and
[WA Ecology](https://fortress.wa.gov/ecy/ezshare/EAP/SalishSea/SalishSeaModelBoundingScenarios.html).
