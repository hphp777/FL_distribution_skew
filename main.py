import multiprocessing
from multiprocessing import process
import time
from pipelining import Loader
from torch.utils.data import DataLoader

batch_size = 1

# Define each client's dataloader
client0_dataloader = DataLoader(Loader(0))
client1_dataloader = DataLoader(Loader(1))
client2_dataloader = DataLoader(Loader(2))
client3_dataloader = DataLoader(Loader(3))
client4_dataloader = DataLoader(Loader(4))
client5_dataloader = DataLoader(Loader(5))
client6_dataloader = DataLoader(Loader(6))
client7_dataloader = DataLoader(Loader(7))
client8_dataloader = DataLoader(Loader(8))
client9_dataloader = DataLoader(Loader(9))

for batch, img in enumerate(client0_dataloader):
    pass

# Bring model

# Assign train function to each process
process_list = range(10)
pool = multiprocessing.Pool(processes=10)
# pool.map(, process_list)
pool.close()
pool.join()

# start_time = time.time()
# Measure execution time : time.time() - start_time
