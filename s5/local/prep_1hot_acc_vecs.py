#!/usr/bin/python

import argparse
import csv
import os
import random


def make_acc_1hot(accent_list, random_unk):
    """Convert accent list to 1-hot index vectors in Kaldi text format."""
    with open(accent_list) as accf:
        accents = set(i.strip() for i in accf.readlines())
    if random_unk > 0:
        accents.add('unknown')
    accent_map = {acc: i for i, acc in enumerate(sorted(accents))}
    acc_to_1hot = {}
    for acc, i in accent_map.items():
        acc_to_1hot[acc] = '[{}{}{} ]'.format(
                ''.join(' 0' for m in range(i)),
                ' 1',
                ''.join(' 0' for n in range(len(accent_map) - 1 - i)))
    return acc_to_1hot


def meta_acc_to_1hot(meta_in, acc_to_1hot, random_unk, label_unk):
    """Map utterances in meta file to 1-hot accent vectors."""
    accents = set(acc_to_1hot.keys())
    unk_accs = set()
    utt_to_1hot = {}
    with open(meta_in) as meta_inf:
        meta_reader = csv.DictReader(meta_inf, delimiter='\t')
        for row in meta_reader:
            client_id = row['client_id']
            path = os.path.splitext(row['path'])[0]
            utt_id = '{}-{}'.format(client_id, path)
            if random.random() > random_unk:
                accent = row['accent']
            else:
                accent = 'unknown'
            if accent not in accents:
                if accent not in unk_accs:
                    if label_unk:
                        print('Assigning unexpected accent {} to "unknown"'.format(accent))
                        accent = 'unknown'
                    else:
                        raise RuntimeError('Unexpected accent label: {}\n'.format(accent) +
                        'Remove from meta or add to accent list file {}'.format(args.accent_list))
                unk_accs.add(accent)
            utt_to_1hot[utt_id] = acc_to_1hot[accent]
    return utt_to_1hot


def write_sp_text_ark(utt_to_1hot, out_dir):
    """Write Kaldi 1-hot accent vector text ark for speed-perturbed utts"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # write output in a separate loop cos not sure if kaldi expects
    # speed-perturbed data in a certain order (which default recipe
    # provides) and would complain if they were interleaved
    with open(os.path.join(out_dir, 'accent_vec.txt'), 'w') as outf:
        for speed in [1, 0.9, 1.1]:
            # this sort matches what kaldi uses based on our utt_ids
            for utt, acc in sorted(utt_to_1hot.items()):
                if speed != 1:
                    utt = 'sp{}-{}'.format(speed, utt)
                outf.write('{} {}\n'.format(utt, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare 1-hot accent '
            'vectors for a set of utterances with accent labels, output '
            'to Kaldi text vector ark format.')
    parser.add_argument('--meta_in', required=True, help='Metadata file '
            'with speaker/utterance IDs and accent labels')
    parser.add_argument('--out_dir', required=True, help='Output '
            'directory to write accent vector files')
    parser.add_argument('--accent_list', required=True, help='File listing '
            'all accent labels to enumerate for 1-hot accent vectors')
    parser.add_argument('--label_unk', required=False, action='store_true',
            help='Flag to assign accent labels missing from accent_list file '
            'to "unknown" category if found')
    parser.add_argument('--random_unk', required=False, type=float, default=0.0,
            help='Proportion of utterances to assign to "unknown" accent label')
    parser.add_argument('--seed', required=False, type=int, default=42,
            help='Random seed for sampling utterances into "unknown" accent')
    args = parser.parse_args()

    random.seed(args.seed)

    acc_to_1hot = make_acc_1hot(args.accent_list, args.random_unk)
    utt_to_1hot = meta_acc_to_1hot(args.meta_in, acc_to_1hot, args.random_unk, args.label_unk)
    write_sp_text_ark(utt_to_1hot, args.out_dir)

