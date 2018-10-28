import numpy as np

def next_batch(x, y, batch_size):
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)
    x_shuffle = x[idx]
    y_shuffle = y[idx]
    return x_shuffle[0:batch_size], y_shuffle[0:batch_size]

if __name__ == "__main__":
    batch_size = 3
    x = np.asarray([8,5,0,4,2,3,1,7,6,6,3,2,1,8,1,7,7,8,1,8])
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)

    sequence = []
    start = 0
    for i in range(round(len(x)/batch_size)):
        end = start + batch_size
        sequence.append("%s:%s" % (start,end))
        start = end
    for i in sequence:
        print(idx[int(i[0]):int(i[2:100])])

    # print(idx[0:batch_size], idx[batch_size:2*batch_size], idx[2*batch_size:3*batch_size], idx[3*batch_size:4*batch_size])