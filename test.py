import torch

def test_version():
    a = torch.rand(2, 3)
    print(a._version)
    a[0][0] = 0.0
    b = a.tolist()

    print(b)


if __name__ == "__main__":
    test_version()