model_dir = '../model_data'
data_dir = '../data/balanced_100K'
layer_orders = ['conv-bn-relu', 'conv-relu-bn', 'relu-conv-bn', 'relu-bn-conv', 'bn-conv-relu', 'bn-relu-conv']
filter_sizes = [128]
kernel_sizes = [3]
embedding_sizes = [500]
activations = ['relu']

#-- Train Model --#
with open('train_shallow_balanced_100K.sh', 'w') as output:
  for layer_order in layer_orders:
    for filter_size in filter_sizes:
      for kernel_size in kernel_sizes:
        for embedding_size in embedding_sizes:
          for activation in activations:
            output.write('python train_shallow_balanced_100K.py -m ' + model_dir + ' -d ' + data_dir + ' -l ' + 
              str(layer_order) + ' -f ' + str(filter_size) + ' -k ' + str(kernel_size) + 
              ' -e ' + str(embedding_size) + ' -a ' + str(activation) + ';\n')