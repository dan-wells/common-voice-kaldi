#!/bin/bash

# Recipe for Mozilla Common Voice corpus v2
#
# Copyright 2017   Ewald Enzinger
# Copyright 2020   Dan Wells
# Apache 2.0

corpus=/mnt/data/common_voice_en_1488h_20191210
corpus_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz
accent_only=true
bg_decode=false
use_gpu=false

. ./cmd.sh
. ./path.sh

stage=0

. ./utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  mkdir -p $corpus
  local/download_and_untar.sh $corpus $corpus_url
fi

if [ $stage -le 1 ]; then
  # Select only utterances with accent labels and resample data,
  # excluding New Zealand and India
  if [ $accent_only = true ]; then
    local/prep_accent_corpus.sh en $corpus data
  else
    mkdir -p data/en
    cp $corpus/{train,dev,test}.tsv data/en
  fi
  # clean punctuation, normalise case, remove empty audios/transcripts
  for part in train dev test; do
    local/clean_corpus.py --meta data/en/${part}.tsv --clean data/${part}.tsv
  done
fi

if [ $stage -le 2 ]; then
  # generate kaldi data files
  for part in train dev test; do
    local/data_prep.pl $corpus data $part data/$part
  done
  
  # Prepare ARPA LM and vocabulary using SRILM
  local/prepare_lm.sh data/train
  # Prepare the lexicon and various phone lists
  # Pronunciations for OOV words are obtained using a pre-trained Sequitur model
  local/prepare_dict.sh

  # Prepare data/lang and data/local/lang directories
  utils/prepare_lang.sh data/local/dict '<unk>' data/local/lang data/lang

  utils/format_lm.sh data/lang data/local/lm.gz data/local/dict/lexicon.txt data/lang_test/
fi

if [ $stage -le 3 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  #if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
  #  mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
  #  utils/create_split_dir.pl /export/b{07,14,16,17}/$USER/kaldi-data/mfcc/commonvoice/s5/$mfcc/storage \
  #    $mfccdir/storage
  #fi

  for part in train dev test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 10000 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/subset_data_dir.sh data/train 20000 data/train_20k
fi

# train a monophone system
if [ $stage -le 4 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj 16 --cmd "$train_cmd" \
    data/train_10kshort data/lang exp/mono

  # decode using the mono model
  if [ $bg_decode = true ]; then
    (
      utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph
      for testset in dev; do
        steps/decode.sh --nj 16 --cmd "$decode_cmd" exp/mono/graph \
          data/$testset exp/mono/decode_$testset
      done
    )&
  fi

  steps/align_si.sh --boost-silence 1.25 --nj 8 --cmd "$train_cmd" \
    data/train_20k data/lang exp/mono exp/mono_ali_train_20k
fi

# train a first delta + delta-delta triphone system
if [ $stage -le 5 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_20k data/lang exp/mono_ali_train_20k exp/tri1

  # decode using the tri1 model
  if [ $bg_decode = true ]; then
    (
      utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph
      for testset in dev; do
        steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri1/graph \
          data/$testset exp/tri1/decode_$testset
      done
    )&
  fi

  steps/align_si.sh --nj 8 --cmd "$train_cmd" \
    data/train_20k data/lang exp/tri1 exp/tri1_ali_train_20k
fi

# train an LDA+MLLT system.
if [ $stage -le 6 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_20k data/lang exp/tri1_ali_train_20k exp/tri2b

  # decode using the LDA+MLLT model
  if [ $bg_decode = true ]; then
    (
      utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
      for testset in dev; do
        steps/decode.sh --nj 8 --cmd "$decode_cmd" exp/tri2b/graph \
          data/$testset exp/tri2b/decode_$testset
      done
    )&
  fi

  # Align utts using the tri2b model
  steps/align_si.sh --nj 8 --cmd "$train_cmd" --use-graphs true \
    data/train_20k data/lang exp/tri2b exp/tri2b_ali_train_20k
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 7 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_20k data/lang exp/tri2b_ali_train_20k exp/tri3b

  # decode using the tri3b model
  if [ $bg_decode = true ]; then
    (
      utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph
      for testset in dev; do
        steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
          exp/tri3b/graph data/$testset exp/tri3b/decode_$testset
      done
    )&
  fi
fi

if [ $stage -le 8 ]; then
  # Align utts in the full training set using the tri3b model
  steps/align_fmllr.sh --nj 16 --cmd "$train_cmd" \
    data/train data/lang \
    exp/tri3b exp/tri3b_ali_train

  # train another LDA+MLLT+SAT system on the entire training set
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
    data/train data/lang \
    exp/tri3b_ali_train exp/tri4b

  # decode using the tri4b model
  if [ $bg_decode = true ]; then
    (
      utils/mkgraph.sh data/lang_test exp/tri4b exp/tri4b/graph
      for testset in dev; do
        steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
          exp/tri4b/graph data/$testset \
          exp/tri4b/decode_$testset
      done
    )&
  fi
fi

# Train and extract i-vectors
if [ $stage -le 9 ]; then
  nnet3_affix=""
  local/nnet3/run_ivector_common.sh --stage 0 \
    --train-set train --gmm tri4b --nnet3-affix "$nnet3_affix"
fi

# Train a chain model
if [ $stage -le 10 ]; then
  if [ $use_gpu = true ]; then
    local/chain/run_tdnn.sh --stage 0
  else
    echo "$0: Not expecting to have GPU resources available (use_gpu=$use_gpu)."
    echo "$0: Finishing without training final chain model."
  fi
fi

# Don't finish until all background decoding jobs are finished.
wait
