import os


x_test_path = '/media/nabs/Extra/Projects/MIDS/W266_Project/data/x_test.npy'
y_test_path = '/media/nabs/Extra/Projects/MIDS/W266_Project/data/y_test.npy'
vocab_path = '/media/nabs/Extra/Projects/MIDS/W266_Project/data/full_data/vocabulary_all.pickle'
model_dir = '/media/nabs/Extra/Projects/MIDS/W266_Project/model_data'
model_list = filter(lambda x: '.h5' in x, os.listdir(model_dir))

with open('predict_all.sh', 'w') as output:
	output.write('export CUDA_VISIBLE_DEVICES=5;\n')
	for model_path in model_list:
		output.write('python predict.py -m ' + os.path.join(model_dir,model_path) + ' -x ' + x_test_path + ' -y ' + y_test_path + ' -v ' + vocab_path + ';\n')
