import sys
from time import perf_counter
import datetime
import numpy as np

class Progress:
    """Visual progress message handler"""

    def __init__(self, total_ct, force_log=False, logger=None):
        self._total_ct = total_ct
        self._logger = logger
        self._force_log = force_log
        self.reset()

    def reset(self):
        """Reset the start time"""
        self._start_time = perf_counter()
        self._data_ct = 0

    def skip(self, msgtext):
        """Output a special seeking message indicating no work is being done"""
        msg = f"[seeking to {msgtext}...]"
        if not self._force_log and sys.stdout.isatty():
            sys.stdout.write('\r' + msg)
            sys.stdout.flush()
        else:
            logger.info(msg)

    def update(self, count, data_size=None):
        """Print or log an update message"""
        elapsed = perf_counter() - self._start_time
        to_go = elapsed * (self._total_ct / count - 1)
        total_time = elapsed + to_go
        i_str = (" " * (int(np.log10(self._total_ct)) - int(np.log10(count)))) + str(count)
        pct = str(int(count*100/self._total_ct))
        # Left-pad the percentage
        pct = (" " * (3 - len(pct))) + pct
        msg = f"{i_str}/{self._total_ct} [{pct}%]:  {Progress._format_time(int(elapsed), total_time)} elapsed;  {Progress._format_time(int(to_go), total_time)} to go"
        if data_size is not None:
            self._data_ct += data_size
            rate = int(self._data_ct / elapsed / 1000)
            msg += f";  {rate}KBps"

        if not self._force_log and sys.stdout.isatty():
            sys.stdout.write('\r' + msg)
            sys.stdout.flush()
        else:
            logger.info(msg)

    def finish(self):
        if not self._force_log and sys.stdout.isatty():
            sys.stdout.write('\n')

    @staticmethod
    def _format_time(seconds, rg=None):
        """Nice format of a time in seconds according to an optional range"""
        if rg is None:
            rg = seconds
        if rg > 60:
            if rg > 3600:
                fstr = "%Hh%Mm%Ss"
            else:
                fstr = "%Mm%Ss"
        else:
            fstr = "%Ss"
        dt_mid = datetime.datetime(1990, 1, 5)
        dt = dt_mid + datetime.timedelta(seconds=seconds)
        return dt.time().strftime(fstr)
