if [  $# -lt 3 ] 
  then 
    echo "Usage: extract_wav2vec_features.sh <data> <workingdir> <path_to_wav2vec_model>"
    exit 1
  fi 

. ./path.sh

data=$1
tmpdir=$2
h5dir="$tmpdir/h5"
model=$3
remove_dims=$4

mkdir -p "$h5dir"

wavs=$(awk -F" " '{print $2}' $data/wav.scp)
num_wavs=$(echo "$wavs" | wc -l)
wavs_cleaned=$(for wav in $wavs; do if [ -f $wav ]; then echo "$wav"; fi; done)
num_wavs_cleaned=$(echo "$wavs_cleaned" | wc -l)

if [ $num_wavs_cleaned -lt $num_wavs ]; then 
  diff=$((num_wavs-$num_wavs_cleaned))
  echo "$diff wavs did not exist. These have been removed."
fi;

echo "$wavs_cleaned" | python3 "local/wav2vec_featurize.py" --output $h5dir --use-feat \
  --model "$model" --remove_dims "$4" || exit 1;

out=$tmpdir/wav2vec_feats.tmp

if [ -f $out ]; then rm $out; fi

for file in $(find $tmpdir/h5 -name "*.h5context"); do 
  id=$(basename $file | sed 's/\.h5context//g');

  info=$(h5dump -w 65535  -O /dev/null -y -d info $file); channels=$(echo $info | cut -d' ' -f3 | sed 's/[^0-9]*//g'); frames=$(echo $info | cut -d' ' -f2 | sed 's/[^0-9]*//g'); 
  
  #echo "Converting to HD5 to ARK for $id - Channels: $channels Frames : $frames";  

  formatted=$(h5dump -w 65535  -O /dev/null -y -d features $file | tr -d '\n' | sed 's/,//g' | awk -F' ' "BEGIN { i=1 }; { while ( i <= NF ) { printf \$i; printf \" \"; if ( i > 1 && i % $channels == 0 ) print \"\"; i++ } };")

  nf=$(echo "$formatted" | wc -l)
  if [ $nf -ne $frames ]; then
    echo "Mismatch: expected $frames frames, got $nf. Check your HD5 data?";
    #exit 1;
  fi
  
  echo "$id [" >> $out;
  echo "$formatted ]" >> $out;

done;  

copy-feats "ark:$out" "ark,scp:$tmpdir/feats.ark,$tmpdir/feats.scp"
rm $out
cp $tmpdir/feats.scp $data
