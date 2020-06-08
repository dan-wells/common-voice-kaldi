#!/bin/bash
language=$1
source_dir=$2
out_dir=$3
exclude_accs=$4

set -euo pipefail

mkdir -p $out_dir

if [ ! $(which create-corpora) ]; then
  echo "$0: Mozilla CorporaCreator tool not installed. Get it here to prepare" 
  echo "dataset with accent labels for all utterances: https://github.com/mozilla/CorporaCreator"
  exit 1
fi

# We must repurpose existing metadata files to recreate clips.tsv file
# dumped from Common Voice database expected by CorporaCreator.

echo "$0: creating clips.tsv containing only accent-labelled utterances"
awk -v lang="$language" 'BEGIN {
  FS="\t";
  OFS="\t";
}
# Grab header only from first file
FNR==1 && NR!=1 {
  while (/^client_id/) getline;
}
# Add fields expected by CorporaCreator
NR==1 {
    print $0, "locale", "bucket"
}
# Skip rows with no or accent label "other", or blank transcription
NR>1 && $8!="" && $8!="other" && $3!="" {
    print $0, lang, ""
}' $source_dir/{validated.tsv,invalidated.tsv,other.tsv} > $source_dir/clips.tsv

echo "$0: preparing train-dev-test partitions using CorporaCreator"
create-corpora -d $out_dir -f $source_dir/clips.tsv

# Exclude particular accents from train and dev sets, but keep in test
# TODO(danwells): Add this into create-corpora instead
if [ -n "$exclude_accs" ]; then
  for part in train dev; do
    echo "Excluding utterances from $part meta according to pattern /$exclude_accs/"
    grep -Pv "\t${exclude_accs}$" $out_dir/$language/$part.tsv > tmp.tsv
    mv tmp.tsv $out_dir/$language/$part.tsv
  done
fi

