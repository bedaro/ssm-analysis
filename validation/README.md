Python notebooks for running validations of FVCOM and FVCOM-ICM output
against observations. All of these except the tides rely on a Postgres
database hosting all the observation data; that project is hosted
[elsewhere](https://github.com/bedaro/puget_sound_obsdata).

Just like with populating the database, you need to create a `db.ini` file
in this directory that contains the connection parameters (host, username,
password, ...) to the database.
