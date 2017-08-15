python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o conv-bn-relu -f 64 -k 2 -e 1000 -a relu -l 1;
python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o conv-relu-bn -f 64 -k 2 -e 1000 -a relu -l 1;
python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o relu-conv-bn -f 64 -k 2 -e 1000 -a relu -l 1;
python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o relu-bn-conv -f 64 -k 2 -e 1000 -a relu -l 1;
python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o bn-conv-relu -f 64 -k 2 -e 1000 -a relu -l 1;
python train_multinomial_layer_order_500K.py -m ../model_data -d ../data/multinomial_500K -o bn-relu-conv -f 64 -k 2 -e 1000 -a relu -l 1;
