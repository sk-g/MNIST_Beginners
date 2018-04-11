python MNIST.py --kernel='rbf' --components=100 --pca=0 --max_iter=100 --kernel=linear
python MNIST.py --kernel='rbf' --components=100 --pca=1 --max_iter=100
python MNIST.py --kernel='rbf' --components=100 --pca=1 --max_iter=100 --method=sparse
python MNIST.py --kernel='rbf' --components=100 --pca=1 --max_iter=100 --method=svd
python MNIST.py --kernel='rbf' --components=100 --pca=1 --max_iter=100 --method=kernel
python MNIST.py --kernel='rbf' --components=100 --pca=1 --max_iter=100 --method=incremental
python MNIST.py --kernel='rbf' --components=100 --pca=0 --max_iter=100
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=100
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=100 --method=sparse
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=100 --method=svd
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=100 --method=kernel
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=100 --method=incremental
python MNIST.py --kernel='rbf' --components=500 --pca=1 --max_iter=100
python MNIST.py --kernel='rbf' --components=500 --pca=1 --max_iter=100 --method=sparse
python MNIST.py --kernel='rbf' --components=500 --pca=1 --max_iter=100 --method=svd
python MNIST.py --kernel='rbf' --components=500 --pca=1 --max_iter=100 --method=kernel
python MNIST.py --kernel='rbf' --components=500 --pca=1 --max_iter=100 --method=incremental
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=-1
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=-1 --method=sparse
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=-1 --method=svd
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=-1 --method=kernel
python MNIST.py --kernel='rbf' --components=300 --pca=1 --max_iter=-1 --method=incremental
read -p "press Return to continue"
read -p "press Return to close"