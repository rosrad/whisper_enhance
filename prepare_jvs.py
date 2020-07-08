import json
import argparse
import os
import sys
import re
import os.path as path
import utils
import collections
from concurrent.futures import wait, ProcessPoolExecutor as Executor
# %%


def proc_feat(wav_file, feat_file, force=False):
    utils.ensure_dir_for(feat_file)
    feat_prefix = path.splitext(feat_file)[0]
    sox_cmd = f"sox -r 16000 -b 16 -c 1 {wav_file} {feat_prefix}.wav"
    feat_cmd = "./ahocoder16_64 {0}.wav {0}.f0 {1} {0}.fv".format(feat_prefix, feat_file)
    if not path.exists(f"{feat_prefix}.wav") or force:
        assert utils.run_command(sox_cmd) == 0
    if not path.exists(feat_file) or force:
        assert utils.run_command(feat_cmd) == 0


def proc_corpus(wav_dir, feat_dir):
    feat_list = path.join(feat_dir, "feature.list")
    utils.ensure_dir_for(feat_list)
    with open(feat_list, "w", encoding="utf8") as wf, Executor(max_workers=None) as e:
        fs = []
        for whisper_wav in utils.glob(wav_dir, ".*whisper10/.*.wav", relpath=False, noext=False):
            norm_wav = whisper_wav.replace("whisper10", "parallel100")
            if not path.exists(norm_wav):
                continue
            norm_feat = utils.replace_ext(norm_wav.replace(wav_dir, feat_dir), ".mcc")
            whisper_feat = utils.replace_ext(whisper_wav.replace(wav_dir, feat_dir), ".mcc")
            print(f"{norm_feat} {whisper_feat}", file=wf)
            fs.append(e.submit(proc_feat, norm_wav, norm_feat))
            fs.append(e.submit(proc_feat, whisper_wav, whisper_feat))

        wait(fs)


def main(arg_vec):
    params = {
        "wav_dir": r"/home/boren/data/jvs_ver1;str",
        "feat_dir": r"jvs_ver1_feat;str",

    }
    parser = utils.dict2parser(params)
    args = parser.parse_args(arg_vec)
    print(vars(args))
    proc_corpus(args.wav_dir, args.feat_dir)
    print("All done!")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        arg_vec = sys.argv[1:]
    else:
        params = {
            "wav_dir": r"/home/boren/data/jvs_ver1",
            "feat_dir": r"jvs_ver1_feat",
        }
        arg_vec = utils.dict2arg(params)
    main(arg_vec)

# %%
