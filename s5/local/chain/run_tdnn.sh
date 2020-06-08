#!/bin/bash

# Adapted from egs/mini_librispeech/s5/local/chain/tuning/run_tdnn_1e.sh

# local/chain/compare_wer.sh exp/chain/tdnn1a_sp exp/chain/tdnn1a_sp_online
# System                tdnn1a_sp tdnn1a_sp_online
#WER valid_dev       4.82      4.88
#WER valid_test       4.44      4.27
# Final train prob        -0.0579
# Final valid prob        -0.0718
# Final train prob (xent)   -1.1069
# Final valid prob (xent)   -1.1325

set -euo pipefail

stage=0
#decode_nj=8
train_set=train_sp
test_sets="dev test"
gmm=tri4b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
common_egs_dir=
xent_regularize=0.1

# training options
srand=123
remove_egs=false
reporting_email=

#decode options
# TODO(danwells): assuming accent vector stuff will not work
# with unmodified online decoding scripts
test_online_decoding=false  # if true, it will run the last decoding stage.

# method for passing accent information during training
# options: 1hot, xvec
acc_vec=xvec
acc_vec_affix=1a

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_hires
lores_train_data_dir=data/${train_set}
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 32 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 2 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 4200 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 3 ]; then
  if [ $acc_vec = "1hot" ]; then
    # Set 1-hot vector dim based on accent set seen in training data only
    # -- unseen accents in dev or test will be treated as 'unknown' 
    awk -F'\t' '{ print $8 }' data/train.tsv | \
      sort | uniq | grep -v 'accent' > data/accent_list.txt
    # Prepare 1-hot accent vectors for each utterance, via text ark format
    # Sample 10% of utterances to 'unknown' accent label for test on unseen accents
    for part in $train_set $test_sets; do
      python local/prep_1hot_acc_vecs.py \
        --meta_in data/${part}.tsv \
        --out_dir data/$part/${acc_vec}${acc_vec_affix} \
        --accent_list data/accent_list.txt \
        --random_unk 0.1 \
        --label_unk
      copy-vector ark,t:data/$part/${acc_vec}${acc_vec_affix}/accent_vec.txt \
        ark,scp:data/$part/${acc_vec}${acc_vec_affix}/accent_vec.ark,data/$part/${acc_vec}${acc_vec_affix}/accent_vec.scp
      # Verify vector dim equals accent_list + 1 for unknown
      accent_vec_dim_expected=$(($(cat data/accent_list.txt | wc -l) + 1))
      accent_vec_dim=$(feat-to-dim ark,t:data/$train_set/${acc_vec}${acc_vec_affix}/accent_vec.txt -)
      if [ $accent_vec_dim_expected -ne $accent_vec_dim ]; then
        echo "Mismatched dimensions in 1-hot accent vectors: expected $accent_vec_dim_expected, got $accent_vec_dim"
        exit 1
      fi
    done
  elif [ $acc_vec = "xvec" ]; then
    # nb. this trains xvector extractor (maybe slow) and then extracts them (definitely slow)
    local/xvector/run.sh --xvec-affix $acc_vec_affix --test-sets $test_sets
    for part in $train_set $test_sets; do
      copy-vector scp:data/$part/${acc_vec}${acc_vec_affix}/xvector.scp \
        ark,scp:data/$part/${acc_vec}${acc_vec_affix}/accent_vec.ark,data/$part/${acc_vec}${acc_vec_affix}/accent_vec.scp
      copy-vector ark:data/$part/${acc_vec}${acc_vec_affix}/accent_vec.ark \
        ark,t:data/$part/${acc_vec}${acc_vec_affix}/accent_vec.txt
    done
    accent_vec_dim=$(feat-to-dim ark,t:data/$train_set/${acc_vec}${acc_vec_affix}/accent_vec.txt -)
  fi
fi


if [ $stage -le 4 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$accent_vec_dim name=accent
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0),ReplaceIndex(accent, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=768
  relu-batchnorm-layer name=tdnn2 dim=768 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 dim=768
  relu-batchnorm-layer name=tdnn4 dim=768 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn5 dim=768
  relu-batchnorm-layer name=tdnn6 dim=768 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn7 dim=768 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn8 dim=768 input=Append(-6,-3,0)

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=768 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn8 dim=768 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 5 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --feat.accent-vec-dir=data/$train_set/${acc_vec}${acc_vec_affix} \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --egs.stage=$get_egs_stage \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 6 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test \
    $tree_dir $tree_dir/graph || exit 1;
fi

if [ $stage -le 7 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  # TODO(danwells): accent vector handling only added to
  # nnet3-latgen-faster-parallel so far => decode on CPU
  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          --accent-vec-dir data/$data/${acc_vec}${acc_vec_affix} \
          $tree_dir/graph data/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 8 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph data/${data} ${dir}_online/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
