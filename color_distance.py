import tensorflow as tf
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

def palette_perc(img, clusters: int, verbose=False):
    # img = cv2.imread(img_path)
#     img = img.numpy() 
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dim = (500, 300) #set dim to whatever here
    img = tf.image.resize(img, dim)

    clt = KMeans(n_clusters=clusters)
    k_cluster = clt.fit(tf.reshape(img, [-1, 3]))
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_) # count how many pixels per cluster
    perc = np.zeros(clusters)
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    # perc = dict(sorted(perc.items()))
    
    #for logging purposes
    if verbose:
        print(perc)
        print(k_cluster.cluster_centers_)
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_): 
        palette[:, step:int(step + perc[idx]*width+1), :] = centers
        step += int(perc[idx]*width+1)
    #visualize results:
    if verbose:
        show_img_compar(img, palette)
        
    return perc, k_cluster.cluster_centers_


def calc_color_distance(image,camo_gen):
  '''Take the colors in generated camo and calculate their distance 
  from the primary colors in the landscape
  image :: landscape image 
  camo_gen :: camo from generator
  returns :: scalar quantity representing total distance 
  '''

  #generate the colors and proportions for landscape & camo
  prop_i, colors_i = palette_perc(image, 5, verbose=False)
  prop_c, colors_c = palette_perc(camo_gen, 5, verbose=False)

  #sort the colors by represented proportion in image/camo
  sorted_idxs_i = np.argsort(prop_i)
  sort_ci = colors_i[sorted_idxs_i]
  
  sorted_idxs_c = np.argsort(prop_c)
  sort_cc = colors_c[sorted_idxs_c]
#   sort_ci = [x for _, x in sorted(zip(prop_i, colors_i))] # this fails when there are two elements of equal value in prop_i
#   sort_cc = [x for _, x in sorted(zip(prop_c, colors_c))]

  #get minimum distance between all colors based on proportions 
  cam_list = []
  for i,color in enumerate(sort_ci):
    distance = np.linalg.norm(color - sort_cc[i]) 
    cam_list.append(distance) 
    
  return np.mean(cam_list)