# Author: Javier Duarte
# https://gist.github.com/jmduarte/7adad868dd59ede56dfa8b203bad592e

# testing files
wget https://opendata.cern.ch/record/12102/files/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_test_root_file_index.txt
mkdir -p HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test
for SRCFILE in `cat HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_test_root_file_index.txt`; do
    DSTFILE=HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/test/`basename $SRCFILE`
    if [ ! -f "$DSTFILE" ]; then
    	echo xrdcp $SRCFILE $DSTFILE
	xrdcp $SRCFILE $DSTFILE
    fi
done

# training files
wget https://opendata.cern.ch/record/12102/files/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_root_file_index.txt
mkdir -p HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train
for SRCFILE in `cat HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC_train_root_file_index.txt`; do
    DSTFILE=HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/train/`basename $SRCFILE`
    if [ ! -f "$DSTFILE" ]; then
    	echo xrdcp $SRCFILE $DSTFILE
	xrdcp $SRCFILE $DSTFILE
    fi
