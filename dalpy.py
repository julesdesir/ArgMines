import numpy as np
import os

def convolve2D(f, kernel):
    """
    Convolves a 2D numpy array with a 2D kernel.
    vectorized function
    care using odd kernel size
    """
    # Pad the input array
    if kernel.shape[0]%2 == 0 or kernel.shape[1]%2 == 0:
        raise ValueError('kernel size must be odd')
    pad_width = [(k // 2, k // 2) for k in kernel.shape]
    f_padded = np.pad(f, pad_width, mode='constant')

    # Perform the convolution
    windows = np.lib.stride_tricks.sliding_window_view(f_padded, kernel.shape)
    result = np.einsum('ij,klij->kl', kernel, windows)

    # Adjust the result to match the 'same' mode behavior
    if f.shape != result.shape:
        diff = np.subtract(f.shape, result.shape)
        result = np.pad(result, [(0, diff[0]), (0, diff[1])], mode='constant')

    return result

def zeroPaddingGray(image, n):
    """
    Upsample an image by zero padding
    """
    tf = np.fft.fft2(image)
    tf = np.fft.fftshift(tf)
    tf = np.pad(tf, ((n*image.shape[0]//2, n*image.shape[0]//2), (n*image.shape[1]//2, n*image.shape[1]//2)), mode='constant')
    tf = np.fft.ifftshift(tf)
    padded = np.fft.ifft2(tf)
    return padded/np.max(np.abs(padded))

def zeroPadding(image, n):
    """
    Upsample an image by zero padding
    """
    padded = np.zeros((image.shape[0]*(n+1), image.shape[1]*(n+1), image.shape[2]), dtype=np.complex64)
    for i in range(image.shape[2]):
        padded[:,:,i] = zeroPaddingGray(image[:,:,i], n)
    return padded

class randomkNN:

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.n = len(data)
        self.m = np.sqrt(self.n)

    def addData(self, data, label):
        self.data = np.append(self.data, data, axis=0)
        self.label = np.append(self.label, label, axis=0)   
        self.m = np.sqrt(self.n)

    def predict(self, x, orderedDistance = None):
        U = np.random.rand(self.n)
        labellizingIndex = np.argmin(np.max(np.hstack((self.m * U, np.arange(self.n))), axis=0))
        if orderedDistance is None:
            orderedDistance = np.argsort(np.sum((self.data - x)**2, axis=1))
        labellizingData = orderedDistance[labellizingIndex]
        return self.label[labellizingData]
    

class randomForestkNN:

    def __init__(self, data, label, n_trees=50):
        self.n_trees = n_trees
        self.trees = []
        self.data = data
        self.label = label
        for i in range(n_trees):
            self.trees.append(randomkNN(data, label))

    def addData(self, data, label):
        for i in range(self.n_trees):
            self.trees[i].addData(data, label)

    def predict(self, x):
        predictions = []
        try:
            orderedDistance = np.argsort(np.sum((self.data - x)**2, axis=1))
        except:
            print(type(x[0]), type(self.data[0]))
            print(x, self.data)
        for i in range(self.n_trees):
            predictions.append(self.trees[i].predict(x, orderedDistance))
        return np.mean(predictions)
    
    def toTxtFile(self, filename):
        os.mkdir(filename)
        np.save(os.path.join(filename, "data.npy"), self.data)
        np.save(os.path.join(filename, "label.npy"), self.label)
        with open(os.path.join(filename, "n_trees.txt"), "w") as f:
            f.write(str(self.n_trees))


    def fromTxtFile(self, filename):
        self.data = np.load(os.path.join(filename, "data.npy"))
        self.label = np.load(os.path.join(filename, "label.npy"))
        with open(os.path.join(filename, "n_trees.txt"), "r") as f:
            self.n_trees = int(f.read())
        self.trees = []
        for i in range(self.n_trees):
            self.trees.append(randomkNN(self.data, self.label))


if __name__ == "__main__":
    n = 10000

    x0 = np.random.normal(-np.ones(2),size=(n, 2))
    x1 = np.random.normal(np.ones(2),size=(n, 2))
    x = np.vstack((x0, x1))
    y = np.hstack((np.zeros(n), np.ones(n)))

    # plot the data
    import matplotlib.pyplot as plt

    tree = randomForestkNN(x, y)

    # plot the demarcation line
    x0 = np.linspace(-5, 5, 100)
    x1 = np.linspace(-5, 5, 100)
    x0, x1 = np.meshgrid(x0, x1)
    x0 = x0.reshape(-1)
    x1 = x1.reshape(-1)
    x = np.vstack((x0, x1)).T
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = tree.predict(x[i,:])
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.show()