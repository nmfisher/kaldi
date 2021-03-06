#!/bin/bash

# Copyright 2019 Microsoft Corporation (authors: Xingyu Na)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0
dbase=/virtualmachines/data_w2v/audio_video/fluent_academy/audio/processed
wav2vec="/mnt/external/models/wav2vec_large.pt"
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
  utils/prepare_lang.sh data_w2v/local/dict "<UNK>" data_w2v/local/lang data_w2v/lang || exit 1;
  utils/format_lm.sh data_w2v/lang data_w2v/lm/3gram-mincount/lm_unpruned.gz \
    data_w2v/local/dict/lexicon.txt data_w2v/lang_combined_tg || exit 1;
fi


if [ $stage -le 5 ]; then
  # make features
  local/extract_wav2vec_features.sh data_w2v/train exp/wav2vec/train $wav2vec 
fi

if [ $stage -le 6 ]; then
  local/extract_wav2vec_features.sh data_w2v/test exp/wav2vec/test $wav2vec exp/wav2vec/train/h5/remove_dims 
fi

if [ $stage -le 7 ]; then
  local/train_mono_w2v.sh --nj 1 --cmd "$train_cmd" --totgauss 1200 --num_iters 40 \
    data_w2v/train data_w2v/lang exp/mono_w2v || exit 1;

  utils/mkgraph.sh data_w2v/lang_combined_tg exp/mono_w2v exp/mono_w2v/graph_tg || exit 1;

fi

if [ $stage -le 8 ]; then
  local/decode.sh --cmd "$decode_cmd" --nj 1 \
        exp/mono_w2v/graph_tg data_w2v/test exp/mono_w2v/decode_test_tg || exit 1;
fi

echo "######################"
cat exp/mono_w2v/decode_test_tg/scoring_kaldi/best_cer
cat exp/mono_w2v/decode_test_tg/scoring_kaldi/best_wer
echo "######################"

if [ $stage -le 9 ]; then  
  local/align_si_w2v.sh --boost-silence 1.25 --nj 1 --cmd "$train_cmd" \
    data_w2v/train data_w2v/lang exp/mono_w2v exp/mono_ali_w2v || exit 1;
  local/train_deltas_w2v.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 \
    data_w2v/train data_w2v/lang exp/mono_ali_w2v exp/tri1a_w2v || exit 1;

  utils/mkgraph.sh data_w2v/lang_combined_tg exp/tri1a_w2v exp/tri1a_w2v/graph_tg || exit 1;
fi

if [ $stage -le 10 ]; then
  local/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri1a_w2v/graph_tg data_w2v/test exp/tri1a_w2v/decode_test_tg || exit 1;

fi

echo "######################"
cat exp/tri1a_w2v/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri1a_w2v/decode_test_tg/scoring_kaldi/best_wer
echo "######################"

if [ $stage -le 11 ]; then
  local/align_si_w2v.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
    data_w2v/train data_w2v/lang exp/tri1a_w2v exp/tri1a_280k_ali_w2v || exit 1;
  local/train_deltas_w2v.sh --boost-silence 1.25 --cmd "$train_cmd" 4500 36000 \
    data_w2v/train data_w2v/lang exp/tri1a_280k_ali_w2v exp/tri1b_w2v || exit 1;
fi

if [ $stage -le 12 ]; then
  # test tri1b
  utils/mkgraph.sh data_w2v/lang_combined_tg exp/tri1b_w2v exp/tri1b_w2v/graph_tg || exit 1;
  local/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 1 \
        exp/tri1b_w2v/graph_tg data_w2v/test exp/tri1b_w2v/decode_test_tg || exit 1;
fi
echo "######################"
cat exp/tri1b_w2v/decode_test_tg/scoring_kaldi/best_cer
cat exp/tri1b_w2v/decode_test_tg/scoring_kaldi/best_wer
echo "######################"
exit
if [ $stage -le 13 ]; then
  # train tri2a using train_280k
  steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
    data_w2v/train data_w2v/lang exp/tri1b exp/tri1b_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
    data_w2v/train data_w2v/lang exp/tri1b_280k_ali exp/tri2a || exit 1;
fi

if [ $stage -le 14 ]; then
  # test tri2a
  utils/mkgraph.sh data_w2v/lang_combined_tg exp/tri2a exp/tri2a/graph_tg || exit 1;
#  for c in $test_sets; do
#    (
#      steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
#        exp/tri2a/graph_tg data_w2v/$c/test exp/tri2a/decode_${c}_test_tg || exit 1;
#      if $corpus_lm; then
#        $mkgraph_cmd exp/tri2a/log/mkgraph.$c.log \
#          utils/mkgraph.sh data_w2v/lang_${c}_tg exp/tri2a exp/tri2a/graph_$c || exit 1;
#        steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
#          exp/tri2a/graph_$c data_w2v/$c/test exp/tri2a/decode_${c}_test_clm || exit 1;
#      fi
#    ) &
#  done
#  wait
fi

if [ $stage -le 12 ]; then
  # train tri3a using aidatatang + aishell + primewords + stcmds + thchs (~440k)
#  utils/combine_data.sh data_w2v/train_440k \
#    data_w2v//train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data_w2v/train data_w2v/lang exp/tri2a exp/tri2a_440k_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" 7000 110000 \
    data_w2v/train data_w2v/lang exp/tri2a_440k_ali exp/tri3a || exit 1;
fi

if [ $stage -le 13 ]; then
  # test tri3a
  utils/mkgraph.sh data_w2v/lang_combined_tg exp/tri3a exp/tri3a/graph_tg || exit 1;
#  for c in $test_sets; do
#    (
 #     steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
 #       exp/tri3a/graph_tg data_w2v/$c/test exp/tri3a/decode_${c}_test_tg || exit 1;
 #     if $corpus_lm; then
 #       $mkgraph_cmd exp/tri3a/log/mkgraph.$c.log \
 #         utils/mkgraph.sh data_w2v/lang_${c}_tg exp/tri3a exp/tri3a/graph_$c || exit 1;
 #       steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
 #         exp/tri3a/graph_$c data_w2v/$c/test exp/tri3a/decode_${c}_test_clm || exit 1;
 #     fi
 #   ) &
 # done
  wait
fi

if [ $stage -le 14 ]; then
  # train tri4a using all
  utils/combine_data.sh data_w2v/train_all \
    data_w2v/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;

  steps/align_fmllr.sh --cmd "$train_cmd" --nj 20 \
    data_w2v/train_all data_w2v/lang exp/tri3a exp/tri3a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
    data_w2v/train_all data_w2v/lang exp/tri3a_ali exp/tri4a || exit 1;
fi

if [ $stage -le 15 ]; then
  # test tri4a
  utils/mkgraph.sh data_w2v/lang_combined_tg exp/tri4a exp/tri4a/graph_tg || exit 1;
#  for c in $test_sets; do
#    (
 #     steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
 #       exp/tri4a/graph_tg data_w2v/$c/test exp/tri4a/decode_${c}_test_tg || exit 1;
 #     if $corpus_lm; then
 #       $mkgraph_cmd exp/tri4a/log/mkgraph.$c.log \
 #         utils/mkgraph.sh data_w2v/lang_${c}_tg exp/tri4a exp/tri4a/graph_$c || exit 1;
 #       steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 20 \
 #         exp/tri4a/graph_$c data_w2v/$c/test exp/tri4a/decode_${c}_test_clm || exit 1;
 #     fi
 #   ) &
 # done
 # wait
fi

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
