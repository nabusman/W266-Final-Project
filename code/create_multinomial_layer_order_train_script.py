model_dir = '../model_data'
data_dir = '../data/multinomial_500K'
layer_orders = ['conv-bn-relu', 'conv-relu-bn', 'relu-conv-bn', 'relu-bn-conv', 'bn-conv-relu', 'bn-relu-conv']
num_conv_layers = [1]
filter_sizes = [64]
kernel_sizes = [2]
embedding_sizes = [1000]
activations = ['relu']

#-- Train Model --#
with open('train_multinomial_layer_order_500K.sh', 'w') as output:
  for conv_layers in num_conv_layers:
	  for layer_order in layer_orders:
	    for filter_size in filter_sizes:
	      for kernel_size in kernel_sizes:
	        for embedding_size in embedding_sizes:
	          for activation in activations:
	            output.write('python train_multinomial_layer_order_500K.py -m ' + model_dir + ' -d ' + data_dir + ' -o ' + 
	              str(layer_order) + ' -f ' + str(filter_size) + ' -k ' + str(kernel_size) + 
	              ' -e ' + str(embedding_size) + ' -a ' + str(activation) + ' -l ' + str(conv_layers) + ';\n')