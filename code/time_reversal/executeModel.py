import datetime
from soundNet import *

print('Training started at: {}'.format(datetime.datetime.now()))

if torch.cuda.is_available:
    DEVICE = torch.device('cuda')
    print('#{} CUDA device(s) available'.format(torch.cuda.device_count()))
else:
    DEVICE = torch.device('cpu')
    print('No CUDA devices available')
print('-' * 20)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = '/beegfs/bva212/fma_large_cqt/'
PATH_FILE = 'checksums'
NUM_SECONDS = 5
BATCH_SIZE = 32
VAL_SPLIT = 0.2
TRANSFORM_PROB  = 0.5
SR = 22050
MODEL_SAVE_PATH = '/beegfs/bva212/capstone/'
EPOCHS = 25
LEARNING_RATE = 0.0005
VERBOSE = True 
PRINT_EVERY = 3
SAVE = True
CHECKPOINT_EVERY = 5

print('Training Stats:')
print(f'#Batch Size: {BATCH_SIZE} | #Epochs: {EPOCHS} | Learning Rate: {LEARNING_RATE}')
if save:
	print(f'Saving model & plots in: {MODEL_SAVE_PATH} every {CHECKPOINT_EVERY} epochs')
print('-' * 20)

start = time.time()
soundnet = soundNet(DEVICE, ROOT_DIR, PATH_FILE, SR, BATCH_SIZE, VAL_SPLIT, TRANSFORM_PROB, NUM_SECONDS)
print(f'time for instantiating sound net object - {time.time() - start}')

soundnet.train(num_epochs= EPOCHS, learning_rate= LEARNING_RATE, print_every= PRINT_EVERY, 
			   verbose= VERBOSE, save = SAVE, model_save_path = MODEL_SAVE_PATH, checkpoint_every = CHECKPOINT_EVERY)

print('')
print('')
print('')

print('Training Completed')