import subprocess
import os.path as path
import os
import sys
import re
from pydoc import locate
import distutils.util
import numpy as np
import json
import math


def argv_to_pairs(argv):
    """Convert a argv list to key value pairs. For example,
    -x 10 -y 20 -z=100
    {x: 10, y: 20, z: 100}
    """

    arg_dict = {}

    i = 0
    while i < len(argv):
        if argv[i].startswith('-'):
            entry = re.sub('-+', '', argv[i])

            items = entry.split('=')  # handle the case '-z=100'
            if len(items) == 2:
                key = items[0]
                value = items[1]
                i += 1
            else:  # handle the case  '-x 10'
                key = entry
                value = argv[i + 1]
                i += 2

            if key in arg_dict:
                raise ValueError(
                    'You cannot specify a key multiple'
                    'times in commandline, {}'.format(key))

            arg_dict[key] = value
        else:
            raise ValueError(
                'argv in wrong format, the key must be started'
                'with - or --, but found {}'.format(argv[i]))

    return arg_dict


def chunk(l, n):
    """Yield successive n chunks from l."""
    chunk_size = math.ceil(len(l) / n)
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def ensure_dir_for(file):
    return ensure_dir(path.dirname(file))


def ensure_dir(dirpath):
    if not dirpath:
        return
    if not path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)


def to_streams(streams, content=""):
    if not content:
        return
    for s in streams:
        s.write(content.strip() + "\n")
        s.flush()


def run_command(command, workdir=None, streams=[sys.stdout], env=None):
    log_cmd = command if isinstance(command, str) else " ".join(command)
    to_streams(streams, log_cmd)
    process = subprocess.Popen(command,
                               cwd=workdir, shell=True, stdout=subprocess.PIPE, env=env)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        to_streams(streams, output.decode("utf-8"))

    rc = process.poll()
    return rc


def line(filepath):
    with open(filepath) as f:
        for l in f:
            l = l.strip()
            if l:
                yield l


def print_args(args):
    print("args:")
    for k, v in args.items():
        print("{0}:{1}".format(k, v))


def context(data, c=0):
    n, _ = data.shape
    idx = np.stack([np.arange(-c, c + 1) + i for i in range(n)])
    idx[np.where(idx < 0)] = 0
    idx[np.where(idx >= n)] = n - 1
    return np.reshape(data[idx, :], (n, -1))


def iglob(root, regex, find_file=True, relpath=True, noext=False):
    for base, dirs, files in os.walk(root):
        contents = files if find_file else dirs
        for c in contents:
            fp = os.path.join(base, c)
            rp = path.relpath(fp, root)
            if re.match(regex, rp):
                out_path = rp if relpath else fp
                if noext:
                    out_path = path.splitext(out_path)[0]
                yield out_path


def glob(root, regex, find_file=True, relpath=True, noext=True):
    return list(iglob(root, regex, find_file, relpath, noext))


def replace_ext(file, ext):
    return path.splitext(file)[0] + ext


def str2bool(val):
    return bool(distutils.util.strtobool(val))


def cast_func(type_str):
    return locate(type_str) if type_str != "bool" else str2bool


def dict2parser(params, description=""):
    import argparse
    parser = argparse.ArgumentParser(description)
    for par, exp in params.items():
        value = exp.split(";")
        if len(value) == 1:
            cast = locate(value[0]) if value[0] != "bool" else str2bool
            parser.add_argument('-' + par, '--' + par, help=par, required=True, type=cast)
        elif len(value) == 2:
            cast = locate(value[1]) if value[1] != "bool" else str2bool
            parser.add_argument('-' + par, '--' + par, help=par, required=False, type=cast, default=value[0])
        else:
            raise RuntimeError("Option parsing error:{}".format(value))
    return parser


def dict2arg(params):
    arg_str = " ".join(["-{} {}".format(k, v) for k, v in params.items()])
    return arg_str.split()


def default_main(params, main_func):
    if len(sys.argv) > 2:
        arg_vec = sys.argv[1:]
    else:
        arg_vec = dict2arg(params)
    main_func(arg_vec)


def quantize_wav(x):
    int16_max = np.iinfo(np.int16).max
    int16_min = np.iinfo(np.int16).min

    if x.dtype.kind == 'f':
        x *= int16_max

    sample_to_clip = np.sum(x > int16_max)
    if sample_to_clip > 0:
        print('Clipping {} samples'.format(sample_to_clip))
    x = np.clip(x, int16_min, int16_max)
    x = x.astype(np.int16)

    return x


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)


def format_json(obj, parse_float=True):
    if parse_float:
        obj = json.loads(json.dumps(obj, cls=NumpyAwareJSONEncoder), parse_float=lambda x: round(float(x), 3))
    return obj


css_style = '''
<style type="text/css" media="screen" style="width:100%">
table, th, td {border:0px solid black;background-color:#eee; padding:10px;}
th {background-color:#C6E2FF;color:black;font-family:Tahoma;font-size:15;text-align:left;}
td {background-color:#fff;font-family:Calibri;font-size:15;text-align:center;}
.table { word-break:break-all;font-family:Microsoft YaHei, serif;}
.badge { display:inline-block;padding:.25em .4em;font-size:75%;font-weight:700;line-height:1;text-align:center;white-space:nowrap;vertical-align:baseline;border-radius:.25rem}
.badge:empty { display:none}
.btn .badge { position:relative;top:-1px}
.badge-pill { padding-right:.6em;padding-left:.6em;border-radius:10rem}
.badge-primary { color:#fff;background-color:#007bff}
.badge-primary[href]:focus,
.badge-primary[href]:hover { color:#fff;text-decoration:none;background-color:#0062cc}
.badge-secondary { color:#fff;background-color:#6c757d}
.badge-secondary[href]:focus,
.badge-secondary[href]:hover { color:#fff;text-decoration:none;background-color:#545b62}
.diff sl { color:#898989;padding:0 0.1em 0 0.3em;font-weight:900;}
.diff sr { color:#898989;padding:0 0.3em 0 0.1em;font-weight:900;}
.diff ra { color:#898989;font-weight:900;}
.diff .badge { font-size:60%;}
edit-ins { background-color:#f182af;padding:0 0.1em;border-radius:4px;}
edit-del { text-decoration-color:#717477;text-decoration-line:line-through;background-color:#abd3ff;padding:0 0.1em;border-radius:4px;}
edit-sub1 { text-decoration-color:#717477;text-decoration-line:line-through;background-color:#aaffaa;padding:0 0.1em;border-radius:4px;}
edit-sub2 { background-color:#ffff16;;padding:0 0.1em;border-radius:4px;}
</style>'''
base_css_style = '''
<style type="text/css" media="screen" style="width:100%">
    .base_table,
    .base_table th,
    .base_table td {
        font-family: "Courier New", Courier, monospace;
        border: 1px solid black;
        font-size: 15;
        background-color: #fff;
        border-collapse: collapse;
    }

    .base_table th {
        padding: 5px;
        font-size: 20;
        background-color: #C6E2FF;
        text-align: center;
    }

    .base_table td {
        padding-left: 5px;
        padding-right: 5px;
        text-align: center;
    }

    .cell_table,
    .cell_table td {
        border: 0px solid;
        border-collapse: collapse;
        text-align: center;
    }

    .cell_table tr:nth-child(3n+2) {
        background: #eee;
    }

    .cell_table tr:nth-child(3n) {
        color: red;
    }

    .cell_table tr:nth-child(3n+1) {
        color: red;
        border-top: 1px dashed blue;
    }

    .cell_table tr:nth-child(1) {
        border-top: 0px solid blue;
    }
</style>
'''
html_tmp = '''<html><head><title>{}</title></head>{}<body>{}</body>{}</html>'''
