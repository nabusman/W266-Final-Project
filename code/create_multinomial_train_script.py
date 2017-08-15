model_dir = '../model_data'
data_dir = '../data/multinomial_500K'
num_conv_layers = [1,2,3]
filter_sizes = [64]
kernel_sizes = [1,3,5]
embedding_sizes = [1000]
activations = ['relu', 'leakyrelu']

#-- Train Model --#
with open('train_multinomial_500K.sh', 'w') as output:
  for conv_layers in num_conv_layers:
    for filter_size in filter_sizes:
      for kernel_size in kernel_sizes:
        for embedding_size in embedding_sizes:
          for activation in activations:
            output.write('python train_multinomial_500K.py -m ' + model_dir + ' -d ' + data_dir + ' -l ' + 
              str(conv_layers) + ' -f ' + str(filter_size) + ' -k ' + str(kernel_size) + 
              ' -e ' + str(embedding_size) + ' -a ' + str(activation) + ';\n')