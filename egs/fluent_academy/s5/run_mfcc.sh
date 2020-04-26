#!/bin/bash

# Copyright 2019 Microsoft Corporation (authors: Xingyu Na)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0
dbase=/virtualmachines/data/audio_video/fluent_academy/audio/processed
corpus_lm=false   # interpolate with corpus lm

. utils/parse_options.sh

if [ $stage -le 1 ]; then
  local/data_prep.sh $dbase data || exit 1;
fi

if [ $stage -le 2 ]; then
  # normalize transcripts
    local/prepare_dict.sh;
fi

if [ $stage -le 3 ]; then
  # train LM using transcription
  local/train_lms.sh || exit 1;
fi

echo "LM trained";

if [ $stage -le 4 ]; then
  # prepare LM
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
  utils/format_lm.sh data/lang data/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_combined_tg || exit 1;
fi


if [ $stage -le 5 ]; then
  # make features
  mfccdir=mfcc
  steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 1 \
    data/train exp/make_mfcc/train $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/train \
    exp/make_mfcc/train $mfccdir || exit 1;
fi

if [ $stage -le 6 ]; then
  mfccdir=mfcc
  steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 1 \
  	data/test exp/make_mfcc/test $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/test \
    exp/make_mfcc/$c/test $mfccdir/$c
fi

if [ $stage -le 7 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj 1 --totgauss 1200  --cmd "$train_cmd" \
    data/train data/lang exp/mono || exit 1;

  utils/mkgraph.sh data/lang_combined_tg exp/mono exp/mono/graph_tg || exit 1;

fi

if [ $stage -le 8 ]; then
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/mono/graph_tg data/test exp/mono/decode_test_tg || exit 1;
fi

echo "###################### RESULTS FOR MONO #################################"
cat exp/mono/decode_test_tg/scoring_kaldi/best_cer
cat exp/mono/decode_test_tg/scoring_kaldi/best_wer
echo "#########################################################################"


if [ $stage -le 9 ]; then  
  steps/align_si.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 \
    data/train data/lang exp/mono_ali exp/tri1a || exit 1;

  utils/mkgraph.sh data/lang_combined_tg exp/tri1a exp/tri1a/graph_tg || exit 1;
fi

if [ $stage -le 10 ]; then
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri1a/graph_tg data/test exp/tri1a/decode_test_tg || exit 1;

fi

echo "###################### RESULTS FOR TRI1A ###############################"
cat exp/tri1a/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri1a/decode_test_tg/scoring_kaldi/best_wer
echo "########################################################################"

if [ $stage -le 11 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
    data/train data/lang exp/tri1a exp/tri1a_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4500 36000 \
    data/train data/lang exp/tri1a_280k_ali exp/tri1b || exit 1;
fi

if [ $stage -le 9 ]; then
  # test tri1b
  utils/mkgraph.sh data/lang_combined_tg exp/tri1b exp/tri1b/graph_tg || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri1b/graph_tg data/test exp/tri1b/decode_test_tg || exit 1;
fi

echo "###################### RESULTS FOR TRI1B ###############################"
cat exp/tri1b/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri1b/decode_test_tg/scoring_kaldi/best_wer
echo "########################################################################"

if [ $stage -le 10 ]; then
  # train tri2a using train_280k
  steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
    data/train data/lang exp/tri1b exp/tri1b_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
    data/train data/lang exp/tri1b_280k_ali exp/tri2a || exit 1;
fi

if [ $stage -le 11 ]; then
  # test tri2a
  utils/mkgraph.sh data/lang_combined_tg exp/tri2a exp/tri2a/graph_tg || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri2a/graph_tg data/test exp/tri2a/decode_test_tg || exit 1;
fi

echo "###################### RESULTS FOR TRI2A ###############################"
cat exp/tri2a/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri2a/decode_test_tg/scoring_kaldi/best_wer
echo "########################################################################"

if [ $stage -le 12 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
    data/train data/lang exp/tri2a exp/tri2a_440k_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" 7000 110000 \
    data/train data/lang exp/tri2a_440k_ali exp/tri3a || exit 1;
fi

if [ $stage -le 13 ]; then
  # test tri3a
  utils/mkgraph.sh data/lang_combined_tg exp/tri3a exp/tri3a/graph_tg || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri3a/graph_tg data/test exp/tri3a/decode_test_tg || exit 1;
  wait
fi

echo "###################### RESULTS FOR TRI3A ###############################"
cat exp/tri3a/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri3a/decode_test_tg/scoring_kaldi/best_wer
echo "########################################################################"

if [ $stage -le 14 ]; then
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 4 \
    data/train data/lang exp/tri3a exp/tri3a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
    data/train data/lang exp/tri3a_ali exp/tri4a || exit 1;
fi

if [ $stage -le 15 ]; then
  # test tri4a
  utils/mkgraph.sh data/lang_combined_tg exp/tri4a exp/tri4a/graph_tg || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri4a/graph_tg data/test exp/tri4a/decode_test_tg || exit 1;
fi

echo "###################### RESULTS FOR TRI4A ###############################"
cat exp/tri4a/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri4a/decode_test_tg/scoring_kaldi/best_wer
echo "########################################################################"

if [ $stage -le 16 ]; then
  # run clean and retrain
  local/run_cleanup_segmentation.sh --test-sets "$test_sets" --corpus-lm $corpus_lm
fi

if [ $stage -le 17 ]; then
  # collect GMM test results
  if ! $corpus_lm; then
    for c in $test_sets; do
      echo "$c test set results"
      for x in exp/*/decode_${c}*_tg; do
        grep WER $x/cer_* | utils/best_wer.sh
      done
      echo ""
    done
  else
    # collect corpus LM results
    for c in $test_sets; do
      echo "$c test set results"
      for x in exp/*/decode_${c}*_clm; do
        grep WER $x/cer_* | utils/best_wer.sh
      done
      echo ""
    done
  fi
fi

exit 0;

# chain modeling script
local/chain/run_cnn_tdnn.sh --test-sets "$test_sets"
for c in $test_sets; do
  for x in exp/chain_cleaned/*/decode_${c}*_tg; do
    grep WER $x/cer_* | utils/best_wer.sh
  done
done
