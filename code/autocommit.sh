export CUDA_VISIBLE_DEVICES=1;
cd /home/nabs/Projects/MIDS/W266_Project/code;
python create_predict_script.py;
source predict_all.sh;
cd /home/nabs/Projects/MIDS/W266_Project/;
git add --all;
git commit -m 'autocommit';
git push origin master;