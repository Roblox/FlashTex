import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import os
import gzip

import argparse

from rich.progress import track

def worker(queue, count, gpu):
    while True:
        item = queue.get()
        if item is None:
            break
            
        obj_path = item['object_path']
        
        save_path = item['save_path']
        
        if os.path.exists(save_path):
            queue.task_done()
            print('========', obj_path, 'rendered', '========')
            continue
        else:
            os.makedirs(save_path, exist_ok = True)
            
        # Perform some operation on the item
        print(obj_path, gpu)
        command = (
            # f"export DISPLAY=:0.{gpu} &&"
            f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" blender -b -P ../blender_script.py --"
            f" --object_path {obj_path}"
            f" --output_dir {save_path}"
            f" --only_northern_hemisphere"
            f" --num_renders 4"
            f" --engine CYCLES"
            f" > /dev/null 2>&1"
        )
        print(command)
        subprocess.run(command, shell=True)
        
        # update the metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_file = json.load(f)

        metadata_file["annotations"] = item
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_file, f, indent=2, sort_keys=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=8)

    args = parser.parse_args()
    
    use_gpus = ['0', '1', '2', '3', '4', '5', '6', '7']
    use_gpus = use_gpus[:args.gpus]
    print(use_gpus)
    
    workers_per_gpu = 4
    
    with open('objaverse/hf-objaverse-v1/object-paths.json') as f:
        object_paths = json.load(f)
    
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    for gpu_i in use_gpus:
        for worker_i in range(workers_per_gpu):
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

            
    object_paths_list = list(object_paths.values())
    object_paths_list.sort()


    read_metadata = None

    for object_path in track(object_paths_list):
        current_metadata = object_path.split('/')[-2]

        if read_metadata != current_metadata:
            read_metadata = current_metadata

            metadata_path = os.path.join('objaverse/hf-objaverse-v1/metadata/', f"{current_metadata}.json.gz")

            with gzip.open(metadata_path, 'rb') as f:
                annotations = json.load(f)

        uid = object_path.split('/')[-1][:-4]

        if not annotations[uid]['staffpickedAt']:
            continue
            
        obj_path = os.path.join('objaverse/hf-objaverse-v1/', object_path)
        
        save_path = os.path.join('render/staffpicked/', obj_path.split('/')[-1][:-4])
        
        if os.path.exists(save_path):
            continue
            
        queue_item = annotations[uid].copy()
        queue_item['object_path'] = obj_path
        queue_item['save_path'] = save_path
        queue.put(queue_item)
        

    # Add sentinels to the queue to stop the worker processes
    for i in range(len(use_gpus) * workers_per_gpu):
        queue.put(None)
        
            
    # Wait for all tasks to be completed
    queue.join()
    

if __name__ == '__main__':
    main()