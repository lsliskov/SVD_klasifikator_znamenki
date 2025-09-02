from __future__ import annotations
# Sve-na-jednom: baze, evaluacija, pretprocesiranje i GUI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageOps, ImageFilter
from tensorflow.keras.datasets import mnist

EPS = 1e-12  # prag za "prazno" platno ili nultu normu

# =========================================
# 1) CORE: baze po klasi + predikcija kutom
# =========================================
def _napravi_indekse_po_klasi(y_train, *, seed=123, use_first=False, broj_klasa=None):
    """
    Vrati dict {klasa: permutirani_indeksi_te_klase} (reproducibilno ako je zadan seed).
    Ako use_first=True, ne permutira.
    """
    y_train = np.asarray(y_train).astype(int)
    if broj_klasa is None:
        broj_klasa = int(np.max(y_train)) + 1
    rng = np.random.default_rng(seed)
    indeksi_po_klasi = {}
    for d in range(broj_klasa):
        idx = np.where(y_train == d)[0]
        if len(idx) == 0:
            raise ValueError(f"Klasa {d} nema uzoraka u trainu.")
        if use_first:
            indeksi_po_klasi[d] = idx
        else:
            perm = idx.copy()
            rng.shuffle(perm)
            indeksi_po_klasi[d] = perm
    return indeksi_po_klasi

def _izgradi_baze_za_k(x_train, indeksi_po_klasi, k, *, rel_tol=1e-2):
    """
    Za svaku klasu uzmi prvih k uzoraka -> SVD(A), A=(dim,k); zadrži r prema rel_tol.
    Vraća: (baze, rangovi) gdje je baze[d] = U_d[:, :r_d].
    """
    baze, rangovi = {}, {}
    for d, idx in indeksi_po_klasi.items():
        if len(idx) < k:
            raise ValueError(f"Klasa {d}: dostupno {len(idx)}, traženo {k}.")
        sel = idx[:k]
        A = x_train[sel].reshape(k, -1).T.astype(np.float32)  # (dim, k)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        if S.size == 0:
            r = 1
        else:
            r = int(np.sum(S > (rel_tol * S[0])))
            r = max(r, 1)
        baze[d] = U[:, :r].astype(np.float32)
        rangovi[d] = r
    return baze, rangovi

def napravi_baze(x_train, y_train, *, k, seed=123, use_first=False, rel_tol=1e-2):
    """
    Wrapper: u jednom potezu izgradi (baze, rangovi) za zadani k.
    """
    indeksi = _napravi_indekse_po_klasi(y_train, seed=seed, use_first=use_first)
    return _izgradi_baze_za_k(x_train, indeksi, k, rel_tol=rel_tol)

def _predikcija_po_kutu(x_vec, baze):
    """
    Vrati (label, theta_rad) gdje je theta kut između x i projekcije na bazu (manji je bolji).
    """
    x = np.asarray(x_vec, dtype=np.float32).reshape(-1)
    nx = np.linalg.norm(x)
    if nx < EPS:
        return -1, np.pi / 2.0
    best_label, best_theta = -1, np.inf
    for label, U in baze.items():
        proj = U @ (U.T @ x)
        c = np.linalg.norm(proj) / (nx + EPS)
        theta = np.arccos(np.clip(c, -1.0, 1.0))
        if theta < best_theta:
            best_theta, best_label = theta, int(label)
    return best_label, float(best_theta)

# ==================================================
# 2) EVALUACIJA + (opcionalna) MATRICA KONFUZIJE
# ==================================================
def evaluiraj_kut(x, y, baze, *, crtaj_matricu=True):
    """
    x : (N, 28, 28) ili (N, 784)
    y : (N,)
    baze : dict[label] = U  (784, r)
    Vraća: (acc, konf, klase, y_pred)
    """
    x = np.asarray(x)
    if x.ndim == 3:
        x = x.reshape(len(x), -1)
    y = np.asarray(y).astype(int)
    y_pred = np.array([_predikcija_po_kutu(x[i], baze)[0] for i in range(len(x))], dtype=int)

    klase = sorted(set(map(int, np.unique(y))) | set(map(int, baze.keys())))
    mapi = {c: i for i, c in enumerate(klase)}
    K = len(klase)
    konf = np.zeros((K, K), dtype=int)
    for t, p in zip(y, y_pred):
        if t in mapi and p in mapi:
            konf[mapi[t], mapi[p]] += 1

    acc = float((y_pred == y).mean())
    print(f"Ukupna točnost: {acc*100:.2f}%")

    if crtaj_matricu:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(konf)
        ax.set_title("Matrica konfuzije")
        ax.set_xlabel("Predikcija")
        ax.set_ylabel("Stvarna klasa")
        ax.set_xticks(np.arange(K), labels=klase)
        ax.set_yticks(np.arange(K), labels=klase)
        for i in range(K):
            for j in range(K):
                v = konf[i, j]
                if v:
                    ax.text(j, i, str(v), ha="center", va="center")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    return acc, konf, klase, y_pred

# ==================================================
# 3) WRAPPER ZA GUI: klasifikacija + kut u stupnjevima
# ==================================================
def klasificiraj_slika_kut(slika, baze):
    """
    Jednostavni wrapper: koristi _predikcija_po_kutu, kut vrati u stupnjevima.
    """
    x = np.asarray(slika, dtype=np.float32).reshape(-1)
    if np.linalg.norm(x) < EPS:
        return -1, 90.0
    label, theta_rad = _predikcija_po_kutu(x, baze)
    return int(label), float(np.degrees(theta_rad))

# ==================================================
# 4) PRETPROCES: crtež -> MNIST-like 28x28 -> vektor
# ==================================================
def to_mnist_like_28x28(img_pil: Image.Image) -> Image.Image:
    """
    Grayscale -> (po potrebi) invert -> crop foreground -> blur -> resize max dim=20 -> centriraj u 28x28 -> centroid shift.
    """
    img = img_pil.convert("L")
    if np.array(img, dtype=np.uint8).mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img, dtype=np.uint8)
    mask = arr > 0
    if not mask.any():
        return Image.new("L", (28, 28), 0)

    ys, xs = np.where(mask)
    crop = Image.fromarray(arr[ys.min():ys.max()+1, xs.min():xs.max()+1])
    crop = crop.filter(ImageFilter.GaussianBlur(radius=0.5))
    w, h = crop.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, int(round(w*scale))), max(1, int(round(h*scale))))
    crop = crop.resize(new_size, Image.BICUBIC)

    canv = np.zeros((28, 28), dtype=np.float32)
    dig = np.array(crop, dtype=np.float32)
    offx = (28 - dig.shape[1]) // 2
    offy = (28 - dig.shape[0]) // 2
    canv[offy:offy+dig.shape[0], offx:offx+dig.shape[1]] = dig

    yy, xx = np.mgrid[0:28, 0:28]
    s = canv.sum()
    if s > 0:
        cy = (yy*canv).sum()/s
        cx = (xx*canv).sum()/s
        dy, dx = 13.5 - cy, 13.5 - cx
        shiftx, shifty = int(round(dx)), int(round(dy))
        shifted = np.zeros_like(canv)
        ysrc0, ydst0 = max(0, -shifty), max(0,  shifty)
        xsrc0, xdst0 = max(0, -shiftx), max(0,  shiftx)
        H = 28 - abs(shifty)
        W = 28 - abs(shiftx)
        if H > 0 and W > 0:
            shifted[ydst0:ydst0+H, xdst0:xdst0+W] = canv[ysrc0:ysrc0+H, xsrc0:xsrc0+W]
        canv = shifted

    return Image.fromarray(np.clip(canv, 0, 255).astype(np.uint8))

def img28_to_vector(img28: Image.Image) -> np.ndarray:
    """
    28x28 PIL -> vektor (784,) float32; po potrebi invert.
    """
    arr = np.array(img28.convert("L"), dtype=np.uint8)
    if arr.mean() > 127:
        arr = 255 - arr
    return arr.reshape(-1).astype(np.float32)

# ==================================================
# 5) GUI ZA CRTANJE I PREDIKCIJU
# ==================================================
def draw_and_predict_gui(baze, scale=10, linewidth=22):
    """
    Lijevo: platno za crtanje; desno: normalizirana 28x28 slika.
    Gumbi: 'Klasificiraj', 'Reset'.
    """
    size = 28 * scale
    img = np.zeros((size, size), dtype=np.float32)

    fig = plt.figure(figsize=(7, 4))
    ax_draw = fig.add_axes([0.05, 0.15, 0.55, 0.8])
    ax_view = fig.add_axes([0.65, 0.30, 0.30, 0.60])
    ax_btn_pred = fig.add_axes([0.65, 0.15, 0.14, 0.08])
    ax_btn_reset = fig.add_axes([0.81, 0.15, 0.14, 0.08])

    im_draw = ax_draw.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_draw.set_title("Crtaj lijevim klikom")
    ax_draw.set_axis_off()

    im_view = ax_view.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_view.set_title("Normalizirano 28×28")
    ax_view.set_axis_off()

    btn_pred = Button(ax_btn_pred, "Klasificiraj")
    btn_reset = Button(ax_btn_reset, "Reset")

    drawing = {"prev": None}

    def _stamp_circle(arr, cx, cy, radius, value=255.0):
        y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        arr[y0:y1, x0:x1][mask] = value

    def on_move(event):
        if event.inaxes == ax_draw and event.button == 1 and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            if drawing["prev"] is None:
                drawing["prev"] = (x, y)
            x0, y0 = drawing["prev"]
            L = max(abs(x - x0), abs(y - y0)) + 1
            xs = np.linspace(x0, x, L).astype(int)
            ys = np.linspace(y0, y, L).astype(int)
            rad = max(1, linewidth // 2)
            for xi, yi in zip(xs, ys):
                _stamp_circle(img, xi, yi, rad, value=255.0)
            drawing["prev"] = (x, y)
            im_draw.set_data(img)
            fig.canvas.draw_idle()

    def on_release(event):
        drawing["prev"] = None

    def do_classify(event=None):
        # Canvas -> 28x28
        pil_big = Image.fromarray(img.astype(np.uint8))
        pil_28 = to_mnist_like_28x28(pil_big)
        im_view.set_data(np.array(pil_28))

        # Vektor + provjera praznine
        x_new = img28_to_vector(pil_28)
        if np.linalg.norm(x_new) < EPS:
            fig.suptitle("Platno je prazno — nacrtaj znamenku.", fontsize=12)
            fig.canvas.draw_idle()
            return

        # Predikcija
        pred, theta_deg = klasificiraj_slika_kut(x_new, baze)
        fig.suptitle(f"Predikcija: {pred}  (θ={theta_deg:.2f}°)", x=0.92, y=0.98, ha="right", fontsize=14)
        fig.canvas.draw_idle()

    def do_reset(event=None):
        img[:] = 0.0
        im_draw.set_data(img)
        im_view.set_data(np.zeros((28, 28)))
        fig.suptitle("")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('button_release_event', on_release)
    btn_pred.on_clicked(do_classify)
    btn_reset.on_clicked(do_reset)

    plt.show()

# =========================
# MAIN (primjer pokretanja)
# =========================
if __name__ == "__main__":
    # 1) Učitaj MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)  # vrijednosti 0–255
    x_test  = x_test.astype(np.float32)

    # 2) Izgradi baze po razredu
    k = 100
    baze, rangovi = napravi_baze(x_train, y_train, k=k, seed=42, use_first=False, rel_tol=3e-2)
    print("Rangovi po klasama:", "  ".join(f"{d}:{r}" for d, r in sorted(rangovi.items())))

    # 3) Kratka evaluacija (bez crtanja matrice)
    N = min(1000, len(x_test))
    acc, konf, klase, y_pred = evaluiraj_kut(x_test[:N], y_test[:N], baze, crtaj_matricu=False)
    print(f"Točnost na prvih {N} testnih slika: {acc*100:.2f}%")

    # 4) GUI za crtanje i klasifikaciju
    draw_and_predict_gui(baze, scale=10, linewidth=22)
