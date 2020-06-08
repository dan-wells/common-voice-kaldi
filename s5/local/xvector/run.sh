#!/usr/bin/env bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

. ./cmd.sh
. ./path.sh
set -e

xvec_affix=1a
nnet_dir=exp/xvector_nnet${xvec_affix}
train_set=train_sp
test_sets="dev test"

# data augmentation and filtering to apply
do_augment=false
do_sp=false
do_rvb=false
do_musan=false
do_filter=true

do_eval=true

stage=0

. utils/parse_options.sh


if [ $stage -le 0 ]; then
  data_root=/mnt/data/mozilla_common_voice/en_1488h_2019-12-10
  for part in $train_set $test_sets; do
    utils/copy_data_dir.sh data/${part} data/xvec_${part}
    local/data_prep.pl $data_root data $part data/xvec_${part}
  done
fi


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  # MFCC config is different from that used for ASR, so we keep them separate
  for part in train_sp $test_sets; do
    steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" --write-utt2num-frames true \
      --mfcc-config local/xvector/conf/mfcc.conf data/xvec_${part}
    utils/fix_data_dir.sh data/xvec_${part}
    steps/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
      --vad-config local/xvector/conf/vad.conf data/xvec_${part}
    utils/fix_data_dir.sh data/xvec_${part}
  done
fi


# TODO(danwells): Add working data augmentation, only applied speed perturbation so far
# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ] && [ $do_augment = true ]; then
  if [ $do_sp = true ]; then
    # TODO(danwells): All following steps assume we're reusing train_sp from
    #   preceding model training steps, so this probably won't work if trying
    #   to apply speed perturbation to a different train set
    echo "$0: preparing directory for speed-perturbed data"
    frame_shift=0.01
    awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' \
      data/xvec_${train_set}/utt2num_frames > data/xvec_${train_set}/reco2dur
    utils/data/perturb_data_dir_speed_3way.sh data/xvec_${train_set} data/xvec_${train_set}_sp
  fi

  if [ $do_rvb = true ]; then
    if [ ! -d "RIRS_NOISES" ]; then
      # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
      wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
      unzip rirs_noises.zip
    fi

    # Make a version with reverberated speech
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

    # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
    # additive noise here.
    steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate 8000 \
      data/swbd_sre data/swbd_sre_reverb
    cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
    utils/copy_data_dir.sh --utt-affix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
    rm -rf data/swbd_sre_reverb
    mv data/swbd_sre_reverb.new data/swbd_sre_reverb
  fi

  if [ $do_musan = true ]; then
    # Prepare the MUSAN corpus, which consists of music, speech, and noise
    # suitable for augmentation.
    steps/data/make_musan.sh --sampling-rate 8000 /export/corpora/JHU/musan data

    # Get the duration of the MUSAN recordings.  This will be used by the
    # script augment_data_dir.py.
    for name in speech noise music; do
      utils/data/get_utt2dur.sh data/musan_${name}
      mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
    done

    # Augment with musan_noise
    steps/data/augment_data_dir.py --utt-affix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/swbd_sre data/swbd_sre_noise
    # Augment with musan_music
    steps/data/augment_data_dir.py --utt-affix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/swbd_sre data/swbd_sre_music
    # Augment with musan_speech
    steps/data/augment_data_dir.py --utt-affix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/swbd_sre data/swbd_sre_babble
  fi

  # TODO(danwells): make robust to different combinations of augmentation
  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/swbd_sre_aug data/swbd_sre_reverb data/swbd_sre_noise data/swbd_sre_music data/swbd_sre_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/swbd_sre_aug 128000 data/swbd_sre_aug_128k
  utils/fix_data_dir.sh data/swbd_sre_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" \
    --mfcc-config local/xvector/conf/mfcc.conf data/swbd_sre_aug_128k
  #cp data/train/vad.scp data/train_sp
  steps/compute_vad_decision.sh --nj 16 --cmd "$train_cmd" \
    --vad-config local/xvector/conf/vad.conf data/swbd_sre_aug_128k

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug_128k data/swbd_sre

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
  utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  utils/fix_data_dir.sh data/sre_combined
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/xvector/prepare_feats_for_egs.sh --nj 16 --cmd "$train_cmd" \
    data/xvec_${train_set} data/xvec_${train_set}_no_sil data/xvec_${train_set}_no_sil
  utils/fix_data_dir.sh data/xvec_${train_set}_no_sil

  if [ $do_filter = true ]; then
    # Now, we need to remove features that are too short after removing silence
    # frames.  We want atleast 5s (500 frames) per utterance.
    # (in v2, 300 gives 318735/347136 _sp utts; 200 would be 344829)
    min_len=300
    awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/xvec_${train_set}_no_sil/utt2num_frames > data/xvec_${train_set}_no_sil/utt2num_frames.new
    mv data/xvec_${train_set}_no_sil/utt2num_frames.new data/xvec_${train_set}_no_sil/utt2num_frames
    utils/filter_scp.pl data/xvec_${train_set}_no_sil/utt2num_frames data/xvec_${train_set}_no_sil/utt2spk > data/xvec_${train_set}_no_sil/utt2spk.new
    mv data/xvec_${train_set}_no_sil/utt2spk.new data/xvec_${train_set}_no_sil/utt2spk
    utils/fix_data_dir.sh data/xvec_${train_set}_no_sil

    # TODO(danwells): check statistics on this, maybe don't apply either
    #   -- shouldn't be relevant for LID anyway, remember this was SID first
    # We also want several utterances per speaker. Now we'll throw out speakers
    # with fewer than 8 utterances.
    #min_num_utts=8
    #awk '{print $1, NF-1}' data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2num
    #awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_combined_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
    #mv data/swbd_sre_combined_no_sil/spk2utt.new data/swbd_sre_combined_no_sil/spk2utt
    #utils/spk2utt_to_utt2spk.pl data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/utt2spk

  fi
fi


##### Run x-vector DNN training ######
local/xvector/run_xvector.sh --stage $stage --train-stage -1 \
  --data data/xvec_${train_set}_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs
######################################


if [ $stage -le 7 ]; then
  # Extract per-utterance xvectors for all data
  # - train will be passed as auxiliary inputs during ASR training
  # - others can be used for evaluating xvectors per se
  # NB. this is slow! even on GPU
  for part in $train_set $test_sets; do
    echo "Extracting x-vectors for dataset $part... This will probably take a while!"
    local/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" \
      --use-gpu false --nj 8 \
      $nnet_dir data/xvec_${part} \
      data/xvec_$part/xvec${xvec_affix}
  done
fi


if [ $do_eval = true ]; then
  # TODO(danwells): should centering and/or LDA be applied to
  #   xvectors before passing to ASR model?
  if [ $stage -le 8 ]; then
    # Compute the mean vector for centering the evaluation xvectors.
    $train_cmd data/xvec_$train_set/xvec${xvec_affix}/log/compute_mean.log \
      ivector-mean ark:data/xvec_${train_set}/lang2utt scp:data/xvec_$train_set/xvec${xvec_affix}/xvector.scp \
      ark:data/xvec_$train_set/xvec${xvec_affix}/mean.vec ark,t:data/xvec_$train_set/xvec${xvec_affix}/num_utts.ark || exit 1;
    $train_cmd data/xvec_$train_set/xvec${xvec_affix}/log/compute_global_mean.log \
      ivector-mean scp:data/xvec_$train_set/xvec${xvec_affix}/xvector.scp \
      data/xvec_$train_set/xvec${xvec_affix}/global_mean.vec || exit 1;

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=150
    $train_cmd data/xvec_$train_set/xvec${xvec_affix}/log/lda.log \
      ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
      "ark:ivector-subtract-global-mean scp:data/xvec_$train_set/xvec${xvec_affix}/xvector.scp ark:- |" \
      ark:data/xvec_${train_set}/utt2lang data/xvec_$train_set/xvec${xvec_affix}/transform.mat || exit 1;

    # Train PLDA model.
    $train_cmd data/xvec_$train_set/xvec${xvec_affix}/log/plda.log \
      ivector-compute-plda ark:data/xvec_${train_set}/lang2utt \
      "ark:ivector-subtract-global-mean scp:data/xvec_$train_set/xvec${xvec_affix}/xvector.scp ark:- | transform-vec data/xvec_$train_set/xvec${xvec_affix}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
      data/xvec_$train_set/xvec${xvec_affix}/plda || exit 1;

    # Here we adapt the out-of-domain PLDA model to SRE16 major, a pile
    # of unlabeled in-domain data.  In the future, we will include a clustering
    # based approach for domain adaptation, which tends to work better.
    #$train_cmd exp/xvectors_sre16_major/log/plda_adapt.log \
    #  ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    #  exp/xvectors_sre_combined/plda \
    #  "ark:ivector-subtract-global-mean scp:exp/xvectors_sre16_major/xvector.scp ark:- | transform-vec exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    #  exp/xvectors_sre16_major/plda_adapt || exit 1;
  fi


  if [ $stage -le 9 ]; then
    for part in $test_sets; do
      # Generate AID trials file
      python local/xvector/get_trials.py data/xvec_$part/utt2lang \
        data/xvec_$part/xvec${xvec_affix}/xvector.scp exp/aid_eval/$part/aid_trials.txt

      # Get results using the PLDA model.
      # Average models per language are 'enrolled' using all train data
      # and evaluated against per-utterance models from test
      $train_cmd exp/aid_eval/$part/${xvec_affix}_plda_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        --num-utts=ark:data/xvec_$train_set/xvec${xvec_affix}/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 data/xvec_$train_set/xvec${xvec_affix}/plda - |" \
        "ark:ivector-mean ark:data/xvec_${train_set}/lang2utt scp:data/xvec_$train_set/xvec${xvec_affix}/xvector.scp ark:- | ivector-subtract-global-mean data/xvec_$train_set/xvec${xvec_affix}/global_mean.vec ark:- ark:- | transform-vec data/xvec_$train_set/xvec${xvec_affix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean data/xvec_$train_set/xvec${xvec_affix}/global_mean.vec scp:data/xvec_$part/xvec${xvec_affix}/xvector.scp ark:- | transform-vec data/xvec_$train_set/xvec${xvec_affix}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat exp/aid_eval/$part/aid_trials.txt | cut -d' ' -f1,2 |" exp/aid_eval/$part/xvec${xvec_affix}_plda_scores || exit 1;

      # Evaluate EER, accuracy
      paste exp/aid_eval/$part/aid_trials.txt exp/aid_eval/$part/xvec${xvec_affix}_plda_scores | \
        awk '{print $6,$3}' | compute-eer - | tee exp/aid_eval/$part/xvec${xvec_affix}_plda_eer
      python local/xvector/get_lid_acc.py data/xvec_$part/utt2lang exp/aid_eval/$part/xvec${xvec_affix}_plda_scores | \
        tee exp/aid_eval/$part/xvec${xvec_affix}_plda_acc

      # Evaluate xvector network outputs
      # Temporarily hide extract.config so that running xvector network forward
      # produces log softmax outputs instead of xvectors
      mv $nnet_dir/extract.config $nnet_dir/extract.config.bak
      local/xvector/extract_xvectors.sh --cmd "run.pl --mem 6G" \
        --use-gpu false --nj 8 \
        $nnet_dir data/xvec_$part data/xvec_$part/xvec${xvec_affix}_logsoftmax
      mv $nnet_dir/extract.config.bak $nnet_dir/extract.config
      copy-vector scp:data/xvec_$part/xvec${xvec_affix}_logsoftmax/xvector.scp ark,t:data/xvec_$part/xvec${xvec_affix}_logsoftmax/xvector.txt
      python local/xvector/logsoftmax2acc.py data/xvec_$part/xvec${xvec_affix}_logsoftmax/xvector.txt \
        $nnet_dir/egs/temp/lang2int data/xvec_$part/utt2lang | tee exp/aid_eval/$part/xvec${xvec_affix}_logsoftmax
    done
  fi
fi
