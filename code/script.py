import sys


if __name__ == '__main__':
	writer = open('testfile.txt', 'w')
	writer.write(sys.argv[1] + '\n')