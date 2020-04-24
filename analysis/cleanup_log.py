# Clean up some bugs in log files

import csv
import os
import sys

timestamps = []

relog = []

def cleanup(fname):
    # Parse activity log
    with open(fname, "r") as in_f:
        for row in in_f:
            if row.startswith("object"):
                add_object_row(row)
            if row.startswith("appSwitch"):
                add_switch_row(row)

    with open(fname, "r") as in_f:
        for row in in_f:
            if row.startswith("contextMenu"):
                if get_ts(row) in timestamps:
                    continue
                else:
                    relog.append(row)
            else:
                relog.append(row)

    with open("clean-" + fname, "w") as out_f:
        out_f.writelines(relog)

def add_object_row(r):
    # object, 6.416875, clock, (-0.123; -0.287; -7.937)
    cols = r.split(',')
    timestamps.append(cols[1])

def add_switch_row(r):
    # appSwitch, 40.95583, Lang. Learn, Packer, contextMenu
    cols = r.split(',')
    timestamps.append(cols[1])

def get_ts(r):
    return r.split(',')[1]

def main():
    fname = sys.argv[1]
    cleanup(fname)

if __name__ == '__main__':
    main()