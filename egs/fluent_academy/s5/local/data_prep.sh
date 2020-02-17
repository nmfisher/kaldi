#!/bin/bash

# Copyright 2019 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/stcmds data/stcmds"
  exit 1;
fi

corpus=$1
data=$2

if [ ! -d $corpus ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating train data folder ****"

mkdir -p $data/train
mkdir -p $data/test

num_test=100

# find wav audio file for train

wavs=$(find $corpus -iname "*.wav" | sort -n)
echo "$wavs" | head -n $num_test > $data/test/wav.list
echo "$wavs" | tail -n +$num_test > $data/train/wav.list
n_test=`cat $data/test/wav.list | wc -l`
n_train=`cat $data/train/wav.list | wc -l`
echo "Found $n_train train files, $n_test test files."

for split in train test; do
	cat $data/$split/wav.list | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}' > $data/$split/utt.list
	cat $data/$split/utt.list | awk '{print substr($1,1,3)}' > $data/$split/spk.list
	while read line; do
	  tn=`dirname $line`/`basename $line .wav`.txt;
	  cat $tn;
	done < $data/$split/wav.list > $data/$split/text.list

	paste -d' ' $data/$split/utt.list $data/$split/wav.list > $data/$split/wav.scp
	paste -d' ' $data/$split/utt.list $data/$split/spk.list > $data/$split/utt2spk
	paste -d' ' $data/$split/utt.list $data/$split/text.list |\
	  sed 's/,| |\n//g' |\
	  local/word_segment.py |\
	  tr '[a-z]' '[A-Z]' |\
	  awk '{if (NF > 1) print $0;}' > $data/$split/text

	for file in wav.scp utt2spk text; do
	  sort $data/$split/$file -o $data/$split/$file
	done

	utils/utt2spk_to_spk2utt.pl $data/$split/utt2spk > $data/$split/spk2utt

	rm -r $data/$split/{wav,utt,spk,text}.list

	./utils/validate_data_dir.sh --no-feats $data/$split || exit 1;
done
