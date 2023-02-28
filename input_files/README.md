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

## Working with the `_riv.dat` and `_pnt_wq.dat` files

I made most of these Python notebooks in 2022. In 2023, I created a new
process that allows for unified, easier editing of the separate freshwater
input files for the hydrodynamic and water quality models.

The script `ssm_read_fwinputs.py` will create a multi-sheet Excel file of
all the data from a hydrodynamic rivers file and the corresponding water
quality file. Three worksheets will be created. The first will have the
node list and any comments parsed from the original files, which generally
say what feature the node corresponds to. A third column tracks whether or
not the given source is present in the water quality input file (becuase
not all of Ecology's loadings have water quality constituents). The second
is the vertical distribution of inflow into the node. The third is all of
the flow and water quality data for all nodes at all dates.

```
usage: ssm_read_fwinputs.py [-h] [-s START_DATE] [-v]
                            riv_file pnt_wq_file outfile

Convert freshwater dat files into Excel spreadsheet

positional arguments:
  riv_file              The FVCOM rivers file
  pnt_wq_file           The FVCOM-ICM point source file
  outfile               The destination spreadsheet file

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start-date START_DATE
                        The zero date for the file
  -v, --verbose         Print progress messages during the conversion
```

The script `ssm_write_fwinputs.py` does the reverse: it takes the
spreadsheet and converts it into a pair of input files for the models.

```
usage: ssm_write_fwinputs.py [-h] [-s START_DATE] [-v] [-c COMMENT]
                             infile out_base

Convert a point source discharge spreadsheet into SSM input files

positional arguments:
  infile                The point sources spreadsheet
  out_base              The base of the output filename (_riv.dat and
                        _wq.dat will be appended

optional arguments:
  -h, --help            show this help message and exit
  -s START_DATE, --start-date START_DATE
                        The zero date for the file. Defaults to the earliest
                        date in the file
  -v, --verbose         Print progress messages during the conversion
  -c COMMENT, --comment COMMENT
                        An optional comment to include in the first line
```

There are two sets of tests for these scripts.
`test_ssm_read_fwinputs.py` is just a unit test for key parts of
`ssm_read_fwinputs.py`. `riv_pnt_integration_test.ipynb` is a more complete
integration test of the combined read and write operations to ensure the
data that was read in is written back out again.
