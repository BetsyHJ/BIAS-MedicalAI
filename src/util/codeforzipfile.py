
    # img_ori_to_zip_path = {} # original to zip-path
    # for zip_file_path in image_dirs:
    #     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #         images_zip_path = [x for x in zip_ref.namelist() if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #         for x in images_zip_path:
    #             ori_x = 'CheXpert-v1.0/train/' + '/'.join(x.strip().split('/')[1:])
    #             img_ori_to_zip_path[ori_x] = read_image_from_zip(zip_ref, x)
    zip_ref1 = zipfile.ZipFile(image_dirs[0], 'r')
    zip_ref2 = zipfile.ZipFile(image_dirs[1], 'r')
    zip_ref3 = zipfile.ZipFile(image_dirs[2], 'r')

    img_ori_to_zip_path = {}
    for zip_ref in [zip_ref1, zip_ref2, zip_ref3]:
        images_zip_path = [x for x in zip_ref.namelist() if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for x in images_zip_path:
            ori_x = 'CheXpert-v1.0/train/' + '/'.join(x.strip().split('/')[1:])
            img_ori_to_zip_path[ori_x] = x
    print(datetime.now(), flush=True)

    # Function to read images from the zip file
    def read_image_from_zip(image_path, queue, for_train = True):
        zip_ref = zip_ref1
        if '(train 2)' in image_path:
            zip_ref = zip_ref2
        elif '(train 3)' in image_path:
            zip_ref = zip_ref3
        
        ori_x = 'CheXpert-v1.0/train/' + '/'.join(image_path.strip().split('/')[1:])
        print(image_path, "Loading", flush=True)
        with zip_ref.open(image_path) as image_file:
            if for_train:
                v = transforms[0](PIL_Image.open(image_file).convert('RGB'))
                queue.put({ori_x: v})
            else:
                queue.put({ori_x: transforms[-1](PIL_Image.open(image_file).convert('RGB'))})
    
    zip_file_paths = list(img_ori_to_zip_path.values())

    # # single worker in main process
    # from queue import Queue
    # q = Queue()
    # for i in range(0, 1000):
    #     read_image_from_zip(zip_file_paths[i], q)
    #     q.get()
    # print(datetime.now(), "over")
    # exit(0) # about 2mins

    num_workers = 10
    queue = mp.Queue()  # Create a Queue to collect results
    for i in range(0, 1000, num_workers):
        chunk = zip_file_paths[i:(i+num_workers)]
        processes = []
        for image_path in chunk:
            p = mp.Process(target=read_image_from_zip, args=(image_path, queue, True))
            processes.append(p)
            p.start()
        for p in processes:
            v = queue.get()
            # print(v.keys())
            p.join()
            
        # print(i, datetime.now(), 'end', flush=True)

    
    # # Create a pool of worker processes
    # with Pool(processes=num_workers) as pool:
    #     # Map the function f to the data, this will distribute the work across the workers
    #     results = pool.map(read_image_from_zip, list(img_ori_to_zip_path.values())[1000])
        

    print(datetime.now(), flush=True)
    exit(0)

    train_val_df['image'] = train_val_df['Path'].map(lambda example: read_image_from_zip(img_ori_to_zip_path[example], for_train=True))
    test_df['image'] = test_df['Path'].map(lambda example: read_image_from_zip(img_ori_to_zip_path[example], for_train=False))
    
    zip_ref1.close()
    zip_ref2.close()
    zip_ref3.close()
    print('here', datetime.now(), flush=True)
    exit(0)