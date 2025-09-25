import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)

class Perceptron:
    def __init__(self, lr=0.1, epochs=2000, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.errors_per_epoch = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.w = rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = 1.0 if (np.dot(self.w, xi) + self.b) >= 0 else -1.0
                update = self.lr * (yi - yhat)
                if update != 0:
                    self.w += update * xi
                    self.b += update
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.w + self.b
        return np.where(z >= 0, 1.0, -1.0)

def make_close_separable(n=33, gap=0.9, scale=0.35, seed=200, max_tries=300):
    for s in range(seed, seed + max_tries):
        rng = np.random.default_rng(s)
        n2 = n // 2
        A = rng.normal([0, 0], [scale, scale], size=(n2, 2))
        B = rng.normal([gap, gap], [scale, scale], size=(n - n2, 2))
        X = np.vstack([A, B])
        y = np.hstack([-np.ones(n2), np.ones(n - n2)])
        p = Perceptron(lr=0.1, epochs=2000, random_state=10).fit(X, y)
        if p.errors_per_epoch[-1] == 0:
            return X, y, p, s
    return X, y, p, s  # fallback

# build data and model (n=33, closer clusters)
X1, y1, p1, seed_used = make_close_separable(n=33, gap=0.9, scale=0.35, seed=200)
yhat1 = p1.predict(X1)
acc1 = accuracy(y1, yhat1)
epochs1 = len(p1.errors_per_epoch)
final_errors1 = p1.errors_per_epoch[-1]
print({"acc1": float(acc1), "epochs1": int(epochs1), "final_errors1": int(final_errors1), "seed_used": seed_used})


import matplotlib.pyplot as plt

def plot_points(X, y, ax=None):
    ax = ax or plt.gca()
    ax.scatter(X[y==-1][:,0], X[y==-1][:,1], marker="o", label="-1")
    ax.scatter(X[y==1][:,0], X[y==1][:,1], marker="x", label="+1")
    ax.legend()

def plot_decision_boundary_2d(model, X, y, ax=None, pad=0.5, steps=200):
    ax = ax or plt.gca()
    x_min, x_max = X[:,0].min()-pad, X[:,0].max()+pad
    y_min, y_max = X[:,1].min()-pad, X[:,1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps), np.linspace(y_min, y_max, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.2, levels=[-1,0,1])
    plot_points(X, y, ax=ax)

plt.figure()
plot_decision_boundary_2d(p1, X1, y1)
plt.title("Task 1: Perceptron boundary (n=33, closer clusters)")
plt.savefig("HW2_task1_boundary_close_n33.png", dpi=150, bbox_inches="tight")
plt.close()
