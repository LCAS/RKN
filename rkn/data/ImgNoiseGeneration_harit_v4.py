import numpy as np

def add_img_noise(imgs1_all, first_n_clean, random, r=0.2, s=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
    """
    :param imgs: Images to add noise to
    :param first_n_clean: Keep first_n_images clean to allow the filter to burn in
    :param random: np.random.RandomState used for sampling
    :param (1-r): probability of staying in same noise state
    :param s: probablity of jumping to high noise state
    :param t_ll: lower bound of the interval the lower bound for each sequence is sampled from
    :param t_lu: upper bound of the interval the lower bound for each sequence is sampled from
    :param t_ul: lower bound of the interval the upper bound for each sequence is sampled from
    :param t_uu: upper bound of the interval the upper bound for each sequence is sampled from
    :return: noisy images, factors used to create them
    """
    print("Harit noise gen")

    assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
    if len(imgs1_all.shape) < 5:
        imgs1_all = np.expand_dims(imgs1_all, -1)
        
    batch_size, seq_len = imgs1_all.shape[:2]

    noisy_imgs = np.zeros(imgs1_all.shape, dtype=imgs1_all.dtype)

    for j in range(batch_size):

        imgs1 = imgs1_all[j:j+1]
        imgs = np.swapaxes(imgs1,0,1)

        factors = np.zeros(imgs.shape)

        tp = random.uniform( low = 0, high = 1, size=( seq_len, 1))
        ts = random.uniform( low = 0, high = 1, size=( seq_len, 1))



        tps = np.greater_equal(tp,r).astype(int)
        tph = np.multiply((np.less(ts,s)).astype(int),(np.less(tp,r)).astype(int))
        tpl = np.multiply((np.greater_equal(ts,s)).astype(int),(np.less(tp,r)).astype(int))

        tps_full = np.tile(tps,[imgs1.shape[2],imgs1.shape[3],1,1])

        tps_full_T = np.swapaxes(np.swapaxes(tps_full,1,3),0,2)
        tps_full_expand = np.expand_dims(tps_full_T,axis=4)

        tph_full = np.tile(tph,[imgs1.shape[2],imgs1.shape[3],1,1])
        tph_full_T = np.swapaxes(np.swapaxes(tph_full,1,3),0,2)
        tph_full_expand = np.expand_dims(tph_full_T,axis=4)

        tpl_full = np.tile(tpl,[imgs1.shape[2],imgs1.shape[3],1,1])
        tpl_full_T = np.swapaxes(np.swapaxes(tpl_full,1,3),0,2)
        tpl_full_expand = np.expand_dims(tpl_full_T,axis=4)


        factors[0,:] = random.uniform(low=0.0, high=1.0, size=(imgs.shape[1:]))
        for i in range(seq_len - 1):
            sign = (-1)**random.randint( low=0, high = 2, size=(imgs.shape[1:]))
            low = np.multiply(sign, random.uniform(low=t_ll, high=t_lu, size=(imgs.shape[1:])))
            high = np.multiply(sign, random.uniform(low=t_ul, high=t_uu, size=(imgs.shape[1:])))
            factors[i+1,:] = np.multiply(tps_full_expand[i,:],factors[i,:]) + np.multiply(tpl_full_expand[i,:], low) + np.multiply(tph_full_expand[i,:], high)


        factors[:first_n_clean,:] = 0.0
        noise = np.swapaxes(factors,0,1)

        if imgs.dtype == np.uint8:
            noisy_imgs[j] = (np.clip(noise * 255  + imgs1, a_min=0, a_max=255))
        else:
            noisy_imgs[j] = (np.clip(noise + imgs1, a_min=0.0, a_max=1.0))

    #print('noisy_imgs.shape',noisy_imgs[0].shape) 
    
    #for i in range(batch_size):
    #    noisy_arr=noisy_imgs[i]
    #    buf = "noise_vids/noisy_vid_b_%d_r_%d_s_%d.avi" % (i, math.floor(r*10),math.floor(s*10))
    #    writer = cv2.VideoWriter(buf, cv2.VideoWriter_fourcc(*'PIM1'), 25, (48, 48), False)
    #    for cnt in range(seq_len):
    #        #print('noisy_arr.shape',noisy_arr.shape) 
    #        curr_img=np.squeeze(noisy_arr[cnt,:,:])
    #        print('curr_img.shape',curr_img.shape) 
    #        writer.write(curr_img.astype('uint8'))
    #    print('saving video:',buf) 	
    return np.squeeze(noisy_imgs), None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    random = np.random.RandomState(0)
    # dummy images
    imgs = 0.5*np.ones([200, 150, 1, 1, 1])

    # r smaller -> samples more correlated over time
    # s smaller -> high probabilty of smapling from low noise region
    noisy_imgs, factors = add_img_noise(imgs, first_n_clean=0, random=random,
                               #r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
                               r=0.1, s=0.1, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)

    print('noisy_imgs.shape',np.squeeze(noisy_imgs).shape)
    print('imgs.shape',np.squeeze(imgs).shape)
    print('factors.shape',np.squeeze(factors).shape)
    # Distribution over all sampled factors as histogram
    plt.figure()
    plt.hist(np.ravel(factors), bins=50, normed=True)
    #plt.hist(np.ravel(noisy_imgs-np.squeeze(imgs)), bins=50, normed=True)
    #plt.hist(np.ravel(noisy_imgs-np.squeeze(imgs)), bins=50, normed=True)

    # Factors for first 5 sequences over time
    plt.figure()
    plt.plot(np.squeeze(factors)[:3].T)
    plt.plot((noisy_imgs-np.squeeze(imgs))[1:4].T)
    plt.plot((noisy_imgs)[1:4].T)
    plt.show()
